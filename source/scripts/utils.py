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
from prettytable import PrettyTable
from converters import GID15Converter
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
def eval_model(net, Loader_validation, device, batch_size=1, show_progress=False):
	"""
	Function that evaluates model precision.
	Parameters:
		net (torch network): loaded network
		Loader_validation (Loader)
		device(torch device): device to use
		batch_size (int)
		show_progress (bool): True to show progress bar
	Returns:
		AmIoU (int): average mean intersection over union.
			It's the mIoU averaged on the number of samples.
		AwIoU (int): same as AmIoU but weighted considering the
			support for each class.
		flat confusion matrix (torchmetrics metric): un-normalized
			confusion matrix. Usefull to compute evaluation metrics
		normalized confusion matrix (torchmetrics metric): normalized 
			confusion matrix. Usefull for plotting.
	"""
	macro = 0
	weighted = 0    
	net.eval()	
	# shape: N x H x W (960x224x224)
	normalized_confusion_metric = MulticlassConfusionMatrix(num_classes=16, normalize='true')
	flat_confusion_metric = MulticlassConfusionMatrix(num_classes=16)
	with torch.no_grad():
		for c in range(len(Loader_validation)):									
			dataset = Loader_validation.get_iterable_chunk(c)
			dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
			if show_progress:
				pbar = tqdm(total=len(dataset.chunk_crops)//batch_size, desc=f'Chunk {c+1}')
			for i,(x,y,_,context) in enumerate(dl):								
				x, y = x.to(device), y.to(device)
				if net.requires_context:
					context = context.to(device)
				y_pred = net(x.type(torch.float32), context.type(torch.float32))
				y_pred = y_pred.squeeze().cpu()
				_,pred_mask = torch.max(y_pred, dim=0)
				
				# update global tensor to compute overall mIoU
				p = pred_mask.unsqueeze(0).type(torch.int64).reshape(1,-1)
				t = y.squeeze(1).cpu().type(torch.int64).reshape(1,-1)
				try:
					normalized_confusion_metric.update(p, t)
					flat_confusion_metric.update(p,t)
				except:
					print("Something wrong with updating parameters")
					break
				prediction = pred_mask.cpu().numpy().reshape(-1)
				target = y.cpu().numpy().reshape(-1)
				weighted += jsc(target,prediction, average='weighted') # takes into account label imbalance				
				macro += jsc(target,prediction, average='macro') # simple mean over each class.
				if show_progress:
					pbar.update(1)
			if show_progress:
				pbar.close()
			print("Updating confusion matrix...")
			
	macro_score = macro / ((len(dataset.chunk_crops)*len(Loader_validation))// batch_size)
	weighted_score = weighted / ((len(dataset.chunk_crops)*len(Loader_validation))// batch_size)
	if show_progress:	
		pbar.close()
	# return macro_score, weighted_score, global_mask, global_pred
	return macro_score, weighted_score, flat_confusion_metric, normalized_confusion_metric


def validation_loss(net, Loader_validation, crit, device, bs, show_progress=False):
	loss_values = []
	net.eval()    
	with torch.no_grad():
		for c in range(len(Loader_validation)):
			dataset = Loader_validation.get_iterable_chunk(c)
			dl = torch.utils.data.DataLoader(dataset, batch_size=bs)
			if show_progress:
				pbar = tqdm(total=len(dataset.chunk_crops)//bs, desc=f'Chunk {c+1}')
			for batch_index, (image, index_mask, _, context) in enumerate(dl):
				image, mask = image.to(device), index_mask.to(device)
				if net.requires_context:
					context = context.to(device)
				mask_pred = net(image.type(torch.float32), context.type(torch.float32)).to(device)
				loss = crit(mask_pred, mask.squeeze().type(torch.long))
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
	 

class RandomFlip:
	def __init__(self, prob=0.5):
		self.prob = prob

	def __call__(self, img, mask):
		if random() < self.prob:
			# Apply horizontal flip
			img = T.functional.hflip(img)
			mask = T.functional.hflip(mask)
		if random() < self.prob:
			# Apply vertical flip
			img = T.functional.vflip(img)
			mask = T.functional.vflip(mask)
		return img, mask
	
def inference(net, dataset, indexes, device, converter, mask_only=False):
	net.eval()
	with torch.no_grad():
		for index in indexes:
			image,_, mask = dataset[index]
			image, mask = image.to(device), mask.to(device)
			output = net(image.unsqueeze(0).type(torch.float32))            
			pred_index = torch.argmax(output.squeeze().permute(1,2,0).cpu(), dim=2)         
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
	classes = config['num_classes']
	if netname == 'TSwin':
		return nets.Swin(96,224,classes+1, device).to(device)
	elif netname == 'BSwin':
		return nets.Swin(128,224,classes+1, device).to(device)
	elif netname == 'Unet':
		return nets.Urnet(classes+1).to(device)
	elif netname == 'Fusion':
		return nets.Fusion(classes+1, device).to(device)
	elif netname == 'Unet_torch':
		return nets.UnetTorch(device)
	elif netname == 'Unetv2':
		return nets.Urnetv2(classes+1).to(device)
	elif netname == 'FUnet':
		return nets.FUnet(classes+1).to(device)
	elif netname == 'Resnet101':
		return nets.DeepLabv3Resnet101(classes+1).to(device)
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
		return torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.90, weight_decay=0.0005)
	elif optimizer == 'ADAM1':
		return torch.optim.Adam(net.parameters(), lr=1e-4)
	else:
		raise ValueError("Optimizer name not valid.")

def load_loss(config, device, dataset=None):
	classes = config['num_classes']
	name = config['loss']
	if name == "CEL":
		return nn.CrossEntropyLoss()
	elif name == "DEL":        
		return loss.DiceEntropyLoss(device, classes)
	elif name == "wCEL":
		print("Computing class weights, it might take several minutes...")
		weights = dataset.get_class_weights().to(device)
		return nn.CrossEntropyLoss(weight=weights)
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
	


def precision(confusion, macro=False):
	"""
	returns precision for each class
	"""
	precisions = []
	_, classes = list(confusion.shape)
	for i in range(classes):
		tp = confusion[i,i].item()
		fp = (torch.sum(confusion[:, i])-tp).item()
		if tp == 0:
			precisions.append(0)
		else:
			precisions.append(tp/(tp+fp))
	precisions = torch.tensor(precisions)
	if macro:
		return torch.mean(precisions).item()
	else:
		return precisions

def recall(confusion, macro=False):
	"""
	returns recall for each class
	"""
	recall = []
	_, classes = list(confusion.shape)
	for i in range(classes):
		tp = confusion[i,i].item()
		fn = (torch.sum(confusion[i,:])-tp).item()
		if tp == 0:
			recall.append(0)
		else:				
			recall.append(tp/(tp+fn))
	recall = torch.tensor(recall)
	if macro:
		return torch.mean(recall).item()
	else:
		return recall	


def IoU(confusion, mean=False, exclude_zeros=False):
	_, classes = list(confusion.shape)
	IoU = []
	for i in range(classes):
		tp = confusion[i,i].item()
		fp = (torch.sum(confusion[:, i])-tp).item()
		fn = (torch.sum(confusion[i,:])-tp).item()
		if tp == 0:
			IoU.append(0)
		else:
			IoU.append(tp/(tp+fn+fp))
	IoU = torch.tensor(IoU)
	if exclude_zeros:
		IoU = IoU[IoU.nonzero()].squeeze()
	if mean:
		return torch.mean(IoU).item()
	else:
		return IoU

def F1(confusion, mean=False, exclude_zeros=False):
	_, classes = list(confusion.shape)
	scores = []
	for i in range(classes):
		tp = confusion[i,i].item()
		fp = (torch.sum(confusion[:, i])-tp).item()
		fn = (torch.sum(confusion[i,:])-tp).item()
		if tp == 0:
			scores.append(0)
		else:
			scores.append((2*tp)/(2*tp+fn+fp))
	scores = torch.tensor(scores)
	if exclude_zeros:
		scores = scores[scores.nonzero()].squeeze()
	if mean:
		return torch.mean(scores).item()
	else:
		return scores

def accuracy(confusion):
	_, classes = list(confusion.shape)
	correct_predictions = sum([confusion[i,i].item() for i in range(classes)])
	all_predictions = torch.sum(confusion).item()
	return correct_predictions/all_predictions



def print_metrics(macro, weighted, confusion):
	t = PrettyTable(['Metric', 'Score'])
	t.align = "r"
	t.add_row(['AmIoU', macro])
	t.add_row(['AwIoU', weighted])
	t.add_row(['mIoU', IoU(confusion, mean=True, exclude_zeros=True)])
	t.add_row(['mPrec', precision(confusion, macro=True)])
	t.add_row(['mRec', recall(confusion, macro=True)])
	t.add_row(['Dice', F1(confusion, mean=True, exclude_zeros=True)])
	t.add_row(['OA', accuracy(confusion)])
	print(t)
	iou = PrettyTable(['Class', 'IoU'])
	iou.align = "r"
	values = IoU(confusion, mean=False, exclude_zeros=False).tolist()
	for i,score in enumerate(values):
		iou.add_row([labels[i], score])
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