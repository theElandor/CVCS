import torch
from sklearn.metrics import jaccard_score as jsc
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from random import random
import torchvision.transforms as T
import nets
from dataset import GID15
import numpy as np
import torch.nn as nn
import loss
from PIL import Image
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.segmentation import MeanIoU
from prettytable import PrettyTable
from converters import GID15Converter
import yaml
import torchvision.transforms.v2 as transforms

labels = {
	0:"unlabeled",
	1:"industrial land",
	2:"urban residential",
	3:"rural residential",
	4:"traffic land",
	5:"paddy field",
	6:"irrigated cropland",
	7:"dry cropland",
	8:"garden plot",
	9:"arbor forest",
	10:"shrub land",
	11:"natural grassland",
	12:"artificial grassland",
	13:"river",
	14:"lake",
	15:"pond",
}

def eval_model(net, Loader_validation, device, batch_size=1, show_progress=False, ignore_background=False):
	"""
	Function that evaluates model precision.
	Parameters:
		net (torch network): loaded network
		Loader_validation (Loader)
		device(torch device): device to use
		batch_size (int)
		show_progress (bool): True to show progress bar
	Returns:		
		flat confusion matrix (torchmetrics metric): un-normalized
			confusion matrix. Usefull to compute evaluation metrics
		normalized confusion matrix (torchmetrics metric): normalized 
			confusion matrix. Usefull for plotting.
	""" 
	net.eval()
	# shape: N x H x W (960x224x224)
	ignored_index = 0 if ignore_background else None
	normalized_confusion_metric = MulticlassConfusionMatrix(num_classes=16, normalize='true', ignore_index=ignored_index)
	flat_confusion_metric = MulticlassConfusionMatrix(num_classes=16, ignore_index=ignored_index)	
	with torch.no_grad():
		for c in range(len(Loader_validation)):
			dataset = Loader_validation.get_iterable_chunk(c)
			dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
			if show_progress:
				pbar = tqdm(total=len(dataset.chunk_crops)//batch_size, desc=f'Chunk {c+1}')
			for i,(x,y,_,context) in enumerate(dl):								
				x, y = x.to(device), y.to(device)
				if net.requires_context:context = context.to(device)
				y_pred = net(x.type(torch.float32), context.type(torch.float32)).squeeze().cpu()
				pred_mask = y_pred # if model returns indexes, then no need to torch.max
				if net.returns_logits: _,pred_mask = torch.max(y_pred, dim=0)
				p = pred_mask.unsqueeze(0).type(torch.int64).reshape(1,-1)
				t = y.squeeze(1).cpu().type(torch.int64).reshape(1,-1)				
				normalized_confusion_metric.update(p, t)
				flat_confusion_metric.update(p,t)
				if show_progress:
					pbar.update(1)
			if show_progress:
				pbar.close()
			print("Updating confusion matrix...")
	if show_progress:
		pbar.close()
	# return confusion matrix
	return flat_confusion_metric, normalized_confusion_metric


def validation_loss(net, Loader_validation, crit, device, bs, show_progress=False):
	loss_values = []
	net.eval()    
	with torch.no_grad():
		for c in range(len(Loader_validation)):
			dataset = Loader_validation.get_iterable_chunk(c)
			dl = torch.utils.data.DataLoader(dataset, batch_size=bs)
			if show_progress:
				pbar = tqdm(total=len(dataset.chunk_crops)//bs, desc=f'Chunk {c+1}')
			for image, index_mask, _, context in dl:
				image, mask = image.to(device), index_mask.to(device)
				if net.requires_context:
					context = context.to(device)
				mask_pred = net(image.type(torch.float32), context.type(torch.float32)).to(device)
				loss = crit(mask_pred, mask.squeeze(1).type(torch.long))
				loss_values.append(loss.item())
				if show_progress:
					pbar.update(1)                
			if show_progress:
				pbar.close()
	return loss_values

def save_model(epoch, net, opt,scheduler, train_loss, val_loss, macro_precision, weighted_precision,conf_flat, conf_normalized,batch_size, checkpoint_dir, optimizer):
	torch.save({
		'epoch': epoch,
		'model_state_dict': net.state_dict(),
		'optimizer_state_dict': opt.state_dict(),
		'scheduler_state_dict': scheduler.state_dict(),
		'training_loss_values': train_loss,
		'validation_loss_values': val_loss,
		'batch_size': batch_size,
		'macro_precision': macro_precision, 
		'weighted_precision': weighted_precision,
		'conf_flat': conf_flat,
		'conf_normalized': conf_normalized,
		'optimizer': optimizer,
	}, os.path.join(checkpoint_dir, "checkpoint{}".format(epoch+1)))

	
def inference(net,patch_size,dataset, indexes, device, converter, mask_only=False, border_correction=None):
	crop = T.CenterCrop(patch_size)
	net.eval()
	with torch.no_grad():
		for index in indexes:
			image,mask,context,padded_patch = dataset[index]
			image,mask,context = image.to(device), mask.to(device), context.to(device)
			if border_correction:
				padded_patch.to(device)
				output = net(padded_patch.unsqueeze(0).type(torch.float32))
				output = crop(output)
			else:
				output = net(image.unsqueeze(0).type(torch.float32))
			if net.returns_logits:
				pred_index = torch.argmax(output.squeeze().permute(1,2,0).cpu(), dim=2)
			else:
				pred_index = output
			if not mask_only:
				f, axarr = plt.subplots(1,3)
				axarr[0].imshow(image.permute(1,2,0).cpu())
				axarr[1].imshow(mask.permute(1,2,0).cpu())
				axarr[2].imshow(converter.iconvert(pred_index))
				plt.savefig(os.path.join("output", f"{index}.png"))
			else:
				tensor = converter.iconvert(pred_index)
				arr = ((tensor*255).type(torch.uint8)).numpy()
				image = Image.fromarray(arr)
				image.save(os.path.join("output", f"{index}.png"))
				

def load_network(config, device):
	netname = config['net']
	classes = config['num_classes']+1
	if netname == 'TSwin':
		return nets.Swin(96,224,classes, device).to(device)
	elif netname == 'BSwin':
		return nets.Swin(128,224,classes, device).to(device)
	elif netname == 'Unet':
		return nets.Urnet(classes).to(device)
	elif netname == 'Fusion':
		return nets.Fusion(classes, device).to(device)
	elif netname == 'Unet_torch':
		return nets.UnetTorch(device)
	elif netname == 'Unetv2':
		return nets.Urnetv2(classes).to(device)
	elif netname == 'FUnet':
		return nets.FUnet(classes).to(device)
	elif netname == 'Resnet101':
		return nets.DeepLabv3Resnet101(classes).to(device)
	elif netname == 'MobileNet':
		return nets.DeepLabV3MobileNet(classes).to(device)
	elif netname == 'Ensemble':
		try:
			return Ensemble(classes, device, config.get('ensemble_config'))
		except:
			print("Some error occured when loading ensemble!")
			raise Exception
	elif netname == 'SegformerMod':
		return nets.SegformerMod(classes).to(device)
	else:
		print("Invalid network name.")
		raise Exception
	

def load_gaofen(train, validation, test):    
	train_dataset = GID15(train, random_shift=True)
	validation_dataset = GID15(validation)
	test_dataset = GID15(test)
	return train_dataset, validation_dataset, test_dataset

def count_params(net):
	return sum(p.numel() for p in net.parameters() if p.requires_grad)


def load_optimizer(config, net):
	optimizer = config['opt']
	if optimizer == 'SGD1':
		return torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.90, weight_decay=0.0005)
	elif optimizer == 'ADAM1':
		return torch.optim.Adam(net.parameters(), lr=1e-4)
	elif optimizer == 'ADAM2':
		return torch.optim.Adam(net.parameters(), lr=0.00006)
	else:
		raise ValueError("Optimizer name not valid.")

def load_loss(config, device, dataset=None):
	classes = config['num_classes']+1
	name = config['loss']	
	ignore_background = config['ignore_background']
	
	ignore_index = 0 if ignore_background else -100 # -100 is default
	if name == "CEL":
		return nn.CrossEntropyLoss(ignore_index=ignore_index)
	elif name == "wCEL":
		print("Computing class weights, it might take several minutes...")
		weights = dataset.get_class_weights(classes)
		t = PrettyTable(['Class', 'Weight'])
		for i,score in enumerate(weights):
			t.add_row([labels[i], score.item()])
		print(t, flush=True)
		return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
	else:
		raise Exception

def load_dataset(config):
	return load_gaofen(config['train'], config['validation'], config['test'])

def custom_shuffle(dataset):
	tpe = dataset.tiles_per_img
	images = [_ for _ in range(len(dataset)//tpe)]
	np.random.shuffle(images)
	ranges = []
	for image in images:
		r = [_ for _ in range(image*tpe, (image+1)*tpe)]
		np.random.shuffle(r)
		ranges.append(r)    
	final = sum(ranges, [])    
	return final

def load_loader(dataset, config, shuffle, batch_size=-1):
	if batch_size != -1:
		bs = batch_size
	else:        
		bs = config['batch_size']    
	assert dataset.tiles_per_img % bs == 0, "Tiles per image is not divisible by batch size, unexpected behaviour of DataLoader."    
	if shuffle:
		sampler=custom_shuffle(dataset)
	else:
		sampler=None
	loader = loader = torch.utils.data.DataLoader(dataset, batch_size=bs, sampler=sampler, shuffle=True)
	return loader


def load_device(config):
	if config['device'] == 'gpu':
		assert torch.cuda.is_available(), "Notebook is not configured properly!"
		device = 'cuda:0'
	else:
		device = torch.device('cpu')
	print("Training network on {}".format(torch.cuda.get_device_name(device=device)))
	return device

def load_checkpoint(config, net, load_confusion=False):
	if  'load_checkpoint' in config.keys():
	# Load model checkpoint (to resume training)    
		checkpoint = torch.load(config['load_checkpoint'])
		if net.wrapper:
			net.custom_load(checkpoint)			
		else:
			net.load_state_dict(checkpoint['model_state_dict'])
		TL = checkpoint['training_loss_values']
		VL = checkpoint['validation_loss_values']        
		mIoU = checkpoint['macro_precision']
		wIoU = checkpoint['weighted_precision']		
		print("Loaded checkpoint {}".format(config['load_checkpoint']), flush=True)
		if load_confusion:
			flat = checkpoint['conf_flat']
			normalized = checkpoint['conf_normalized']
			return TL, VL, mIoU, wIoU, flat, normalized
		return TL, VL, mIoU, wIoU
	
def precision_formula(tp, fp, fn):
	return tp/(tp+fp)
def precision_ignore_condition(tp, fp, fn):
	return True if tp+fp == 0 else False

def recall_formula(tp, fp, fn):
	return tp/(tp+fn)
def recall_ignore_condition(tp, fp, fn):
	return True if tp+fn == 0 else False

def IoU_formula(tp, fp, fn):
	return tp/(tp+fn+fp)
def IoU_ignore_condition(tp, fp, fn):
	return True if tp+fn == 0 else False

def F1_formula(tp, fp, fn):
	return (2*tp)/(2*tp+fn+fp)

def _get_class_scores(confusion, formula, ignore_condition):
	"""
	Parameters:
		+ confusion (confusion matrix)
		+ formula (function)
	Returns:
		+ scores (torch.tensor): specified score for each class according to formula
		+ excluded (list): list of class not found in the target (no samples)
	"""
	scores = []
	excluded = []
	_, classes = list(confusion.shape)
	for i in range(classes):
		tp = confusion[i,i].item()
		fp = (torch.sum(confusion[:, i])-tp).item()
		fn = (torch.sum(confusion[i, :])-tp).item()
		if ignore_condition(tp, fp, fn):
			scores.append(0)
			excluded.append(i)
		else:
			scores.append(formula(tp, fp, fn))
	scores = torch.tensor(scores)
	return scores, excluded

def _get_mean_excluding_nptargets(scores, excluded):
	included_precisions = torch.tensor([x for i,x in enumerate(scores) if i not in excluded])
	m = torch.mean(included_precisions).item()
	return m

def _score_wrapper(confusion, formula, ignore_condition, macro, return_excluded):
	scores, excluded = _get_class_scores(confusion, formula, ignore_condition)
	m = _get_mean_excluding_nptargets(scores, excluded)	
	if macro:		
		return (m, excluded) if return_excluded else m
	else:
		return (scores, excluded) if return_excluded else m
	

def precision(confusion, macro=False, return_excluded=False):
	return _score_wrapper(confusion, precision_formula, precision_ignore_condition ,macro, return_excluded)

def recall(confusion, macro=False, return_excluded=False):
	return _score_wrapper(confusion, recall_formula, recall_ignore_condition, macro, return_excluded)

def IoU(confusion, mean=False, return_excluded=False):
	return _score_wrapper(confusion, IoU_formula, IoU_ignore_condition, mean, return_excluded)

def F1(confusion, mean=False, return_excluded=False):
	return _score_wrapper(confusion, F1_formula, IoU_ignore_condition, mean, return_excluded)	

def accuracy(confusion):
	_, classes = list(confusion.shape)
	correct_predictions = sum([confusion[i,i].item() for i in range(classes)])
	all_predictions = torch.sum(confusion).item()
	return correct_predictions/all_predictions

def print_metrics(confusion):
	t = PrettyTable(['Metric', 'Score'])
	t.align = "r"
	t.add_row(['mIoU', IoU(confusion, mean=True)])
	t.add_row(['mPrec', precision(confusion, macro=True)])
	t.add_row(['mRec', recall(confusion, macro=True)])
	t.add_row(['Dice', F1(confusion, mean=True)])
	t.add_row(['OA', accuracy(confusion)])
	print(t)
	iou = PrettyTable(['Class', 'IoU'])
	iou.align = "r"
	values, excluded = IoU(confusion, mean=False, return_excluded=True)
	for i,score in enumerate(values.tolist()):
		iou.add_row([labels[i], score])
	print(f"Excluded classes (not in target): {[el for el in excluded]}")
	print(iou, flush=True)	

def display_configs(configs):
	t = PrettyTable(['Name', 'Value'])
	t.align = "r"
	for key,value in configs.items():
		t.add_row([key, value])
	print(t, flush=True)


def plot_confusion(normalized, path=None):
	fig_, ax_ = normalized.plot(labels = labels.values())
	fig_.set_size_inches(18.5, 10.5)
	if path == None:
		plt.show()
	else:
		plt.savefig(path)

def plot_priors(confusion, sorted=True, path=None):
	c = GID15Converter()	
	label_to_color = {v: k for k, v in c.color_to_label.items()}
	support = torch.sum(confusion, dim=1)
	totals = support.tolist()
	tot = torch.sum(support).item()
	support = [(index, val.item()/tot) for index, val in enumerate(support)]
	if sorted:
		support.sort(key=lambda a: a[1])
		totals.sort()
	fig, ax = plt.subplots()
	fig.set_size_inches(18.5, 10.5)    
	y_pos = np.arange(len(support))
	colors = [(label_to_color[x[0]][0]/255, label_to_color[x[0]][1]/255, label_to_color[x[0]][2]/255) for x in support]
	ax.barh(y_pos, [x[1] for x in support], align='center', color=colors)
	ax.set_yticks(y_pos, labels=[labels[i[0]] for i in support])
	ax.set_xlabel('Class prior')
	ax.set_title('Pixels per class')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	rects = ax.patches	
	for rect, label in zip(rects, totals):
    # Get X and Y placement of label from rect
		x_value = rect.get_width()
		y_value = rect.get_y() + rect.get_height() / 2		
		space = 3
		label = '{:,.2f}M'.format(label/1e6)
		plt.annotate(
			label,
			(x_value, y_value),
			xytext=(space, 0),
			textcoords='offset points', 
			va='center',
			ha='left',
			color = 'black')  
	if path == None:
		plt.show()
	else:
		plt.savefig(path,bbox_inches='tight',dpi=100)

class Ensemble(nn.Module):
	def __init__(self, num_classes, device, config_file):
		super(Ensemble, self).__init__()
		if not config_file:
			print("To use the ensemble you have to specify a config file.")
			print("Add the 'ensemble_config' entry in your evaluation configuration file.")
			raise Exception
		self.requires_context = False
		self.num_classes = num_classes
		self.wrapper = False
		self.returns_logits = False
		self.config_path = os.path.abspath('configs/ensemble/')
		self.config_file = config_file		
		self.device = device
		self.__init_models()

	def __init_models(self):
		self.models = []
		with open(os.path.join(self.config_path, self.config_file), 'r') as file:
			checkpoints = yaml.safe_load(file)
			for key, value in checkpoints.items():
				config = {'net':key, 'load_checkpoint': value, 'device':'gpu', 'num_classes':15}
				model = load_network(config, self.device).to(self.device)
				load_checkpoint(config,model)
				self.models.append(model)


	def forward(self, x:torch.Tensor, context=None):
		logits = []
		for net in self.models:
			net.eval()
			logits.append(net(x))
		preds = [torch.argmax(logit.squeeze().permute(1,2,0).cpu(), dim=2) for logit in logits]
		stack = torch.stack(tuple(preds), dim=0)
		values, _ = torch.mode(stack, dim=0)
		return values

def load_basic_transforms(config):
	"""
	Paramters:
		config file
	Returns:
		Basic image transformations including color jitter, gaussian blur and random rotation.
		If 'augmentation' is False or not specified in the config file, returns None.
	"""
	if config.get('augmentation'):
		image_transforms = transforms.Compose([
			transforms.ColorJitter(contrast=0.6),
			transforms.GaussianBlur(5, sigma=(0.01, 20.0))
		])
		mask_transforms = transforms.RandomRotation(30)
		return image_transforms, mask_transforms
	return None,None

def debug_plot(e,c,i,image,color_mask, context):
	"""
	Parameters:
		e (int): epoch
		c (int): chunk index
		i (int): batch index
		config (config file)
		image (torch.tensor, dim=B,C,H,W)
		color_mask (torch.tensor, dim=B,C,H,W)
		context (torch.tensor, dim=B,C,H,W)
	Function that at the beginning of every epoch plots the first
	encountered patch with the coresponding color mask and context.
	Made for debug purposes. Only the first image of the batch is taken.
	"""		
	image = image[0]
	color_mask = color_mask[0]
	context = context[0]
	f, axarr = plt.subplots(1,3)
	axarr[0].imshow(image.permute(1,2,0).cpu())
	axarr[1].imshow(color_mask.permute(1,2,0).cpu())
	axarr[2].imshow(context.permute(1,2,0).cpu())
	plt.savefig(os.path.join("output", f"epoch{e}_chunk{c}_index{i}.png"))
