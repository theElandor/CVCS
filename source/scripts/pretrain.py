import torch
from pathlib import Path
from tqdm import tqdm
import utils
import yaml
import sys
import dataset
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms
import random

from prettytable import PrettyTable
import traceback

def block_noise(t):
	c,h,w = t.shape
	gran = 15
	dim = 20
	for _ in range(gran):
		tr = random.randint(0, h-dim)
		tc = random.randint(0, w-dim)
		s = random.randint(0,dim)
		t[: ,tr:tr+s, tc:tc+s] = torch.randint(255, (c,s,s))
	return t

def preprocess(image):
	# B,3,224,224
	B,C,H,W = image.shape
	image_transforms = transforms.Compose([
		transforms.GaussianBlur(5, sigma=(10, 25.0)),
		transforms.ElasticTransform(alpha=60.0),
		#transforms.GaussianNoise(), need torchvision 0.19
	])
	image = image_transforms(image)
	for b in range(B):
		image[b] = block_noise(image[b])
	return image
	

def debug_plot_pretrain(image, image_mod):
	image = image[0]	
	image_mod = image_mod[0]
	f, axarr = plt.subplots(1,2)
	axarr[0].imshow(image.permute(1,2,0).cpu())	
	axarr[1].imshow(image_mod.permute(1,2,0).cpu())		
	plt.show()


def validation_loss_pretrain(net, Loader_validation, crit, device, bs, show_progress=False):
	loss_values = []
	net.eval()    
	with torch.no_grad():
		for c in range(len(Loader_validation)):
			dataset = Loader_validation.get_iterable_chunk(c)
			dl = torch.utils.data.DataLoader(dataset, batch_size=bs)
			if show_progress:
				pbar = tqdm(total=len(dataset.chunk_crops)//bs, desc=f'Chunk {c+1}')
			for image, _, _, _ in dl:
				image = image.to(device)
				image_mod = preprocess(image)
				embedding1 = net(image.type(torch.float32)).to(device)
				embedding2 = net(image_mod.type(torch.float32)).to(device)
				loss = crit(embedding1, embedding2)
				loss_values.append(loss.item())
				if show_progress:
					pbar.update(1)                
			if show_progress:
				pbar.close()
	return loss_values


inFile = sys.argv[1]

with open(inFile,"r") as f:
	config = yaml.load(f, Loader=yaml.FullLoader)
utils.display_configs(config)

image_transforms, mask_transforms = utils.load_basic_transforms(config)

Loader_train = dataset.Loader(config['train'],
							  config['chunk_size'],
							  random_shift=True, 
							  patch_size=config['patch_size'],
							  image_transforms=image_transforms,
							  mask_transforms=mask_transforms,
							  load_context = config['load_context'],
							  load_color_mask = config['load_color_mask'])
Loader_validation = dataset.Loader(config['validation'],
								   config['validation_chunk_size'],
								   patch_size=config['patch_size'],
								   load_context = config['load_context'],
								   load_color_mask = config['load_color_mask'])

if config.get('debug'):
	Loader_train.specify([0,1]) # debug, train on 2 images only
	Loader_validation.specify([0]) # debug, validate on 1 image only

device = utils.load_device(config)

t = PrettyTable(['Name', 'Value'])
try:
	net = utils.load_network(config, device)
	t.add_row(['parameters', utils.count_params(net)])    
except:
	traceback.print_exc()
	print("Error in loading network.")    
	exit(0)

t.add_row(['Patch size',Loader_train.patch_size])
t.add_row(['Tpe',Loader_train.tpi])
t.add_row(['Training patches',len(Loader_train.idxs)*Loader_train.tpi])
t.add_row(['Validation patches', len(Loader_validation.idxs)*Loader_validation.tpi])
print(t, flush=True)

try:
	crit = utils.load_loss(config, device, Loader_train)
except:
	traceback.print_exc()
	print("Error in loading loss module.")
	exit(0)
try:
	opt,scheduler = utils.load_optimizer(config, net)
except:
	print("Error in loading optimizer and scheduler")
	exit(0)


training_loss_values=   [] # store every training loss value
validation_loss_values= [] # store every validation loss value
macro_precision=        [] # store AmIoU after each epoch (DEPRECATED)
weighted_precision=     [] # store AwIoU after each epoch (DEPRECATED)
conf_flat=              [] # store unnormalized confusion matrix after each epoch
conf_normalized=        [] # store normalized confusion matrix after each epoch

if  'load_checkpoint' in config.keys():
	# Load model checkpoint (to resume training)
	checkpoint = torch.load(config['load_checkpoint'])
	if net.wrapper:
		net.custom_load(checkpoint)
	else:
		net.load_state_dict(checkpoint['model_state_dict'])    
	opt.load_state_dict(checkpoint['optimizer_state_dict'])
	try:
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
	except:
		print("Scheduler not found in the checkpoint.")
	last_epoch = checkpoint['epoch']+1
	training_loss_values = checkpoint['training_loss_values']
	validation_loss_values = checkpoint['validation_loss_values']
	config['batch_size'] = checkpoint['batch_size']
	macro_precision = checkpoint['macro_precision'] # (DEPRECATED)
	weighted_precision = checkpoint['weighted_precision'] # (DEPRECATED)
	# try to load confusion matrix, usefull for retrocompatibility for old models.
	try:
		conf_flat = checkpoint['conf_flat']
		conf_normalized = checkpoint['conf_normalized']
	except:
		print("Cannot find confusion matrix in specified checkpoint.")
	print("Loaded checkpoint {}".format(config['load_checkpoint']), flush=True)
else:
	last_epoch = 0

assert Path(config['checkpoint_directory']).is_dir(), "Please provide a valid directory to save checkpoints in."

for epoch in range(last_epoch, config['epochs']):
	print("Started epoch {}".format(epoch+1), flush=True)    
	Loader_train.shuffle() # shuffle full-sized images
	for c in range(len(Loader_train)):        
		# if random_tps is specified, then this chunk will contain patches of random size
		dataset = Loader_train.get_iterable_chunk(c,config.get('random_tps'))
		dl = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size']) 
		if config['verbose']:
			pbar = tqdm(total=len(dataset.patches)//config['batch_size'], desc=f'Epoch {epoch+1}, Chunk {c+1}')
		net.train()
		for batch_index, (image, _, _, _) in enumerate(dl):
			image = image.to(device)
			image_mod = preprocess(image) # need to define utils.mod
			# avoid loading context to GPU if not needed
			if config.get('debug_plot'):
				debug_plot_pretrain(image, image_mod)
			embedding1 = net(image.type(torch.float32)).to(device)
			embedding2 = net(image_mod.type(torch.float32)).to(device)
			loss = crit(embedding1, embedding2)
			training_loss_values.append(loss.item())
			opt.zero_grad()
			loss.backward()
			opt.step()
			if config['verbose']:
				pbar.update(1)
				pbar.set_postfix({'Loss': loss.item()})
		if config['verbose']:
			pbar.close()
	if scheduler:
		scheduler.step()
	dataset = dl = None
	print("Running validation...", flush=True)
	validation_loss_values += validation_loss_pretrain(net, Loader_validation, crit, device, config['batch_size'], show_progress=config['verbose'])

	if (epoch+1) % config['freq'] == 0: # save checkpoint every freq epochs
		utils.save_model(epoch, 
					net, opt, scheduler,
					training_loss_values, validation_loss_values, 
					macro_precision, weighted_precision,
					conf_flat,
					conf_normalized,
					config['batch_size'],
					config['checkpoint_directory'], 
					config['opt']
				)
		print("Saved checkpoint {}".format(epoch+1), flush=True)

print("Training Done!")
print(f"Reached training loss: {training_loss_values[-1]}")
print(f"Reached validation loss: {validation_loss_values[-1]}")
