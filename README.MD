# GID15 Virtual Lab
This repo is an open source virtual laboratory to work on remote sensing
image segmentation. It features easy to edit code and scripts to
load state of the art neural networks and train them on datasets like GID15
https://captain-whu.github.io/GID15/.
This projects aims at keeping the code simple and easy to read, so that even unexpert
students can jump in and modify the code according to their needs.
This virtual laboratory revolves around 4 main files:
1) train.py: this files is the entry point if you want to customize your training procedure;
2) utils.py: just a file with some utility functions;
3) nets.py: file where you can declare and add new networks;
3) inference.py: script used to produce segmentation masks once you trained your model.
4) evaluation.py: script used to evaluate the performance of your model.
# Quickstart
In this tutorial we show how to setup the training of a simple network.
You will also learn how to create a config file to customize hyperparameters and other settings. If you are just interested in launching a sample train jump [here](#3-write-the-training-config).

## 1. Write the model
```python
class Unet(nn.Module):
      # classic Unet with some reshape and cropping to match our needs.
	def __init__(self, num_classes):
		super(Unet, self).__init__()		
		self.requires_context = False
		self.wrapper = False
		self.returns_logits = True
    	# encoding part of the Unet vanilla architecture
		self.encode1 = nn.Sequential(
			UnetEncodeLayer(3, 64, padding=1),
			UnetEncodeLayer(64, 64, padding=1), ## keep dimensions unchanged
		)
		self.encode2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			UnetEncodeLayer(64, 128, padding=1),
			UnetEncodeLayer(128, 128, padding=1),
		)
		self.encode3 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			UnetEncodeLayer(128, 256, padding=1),
			UnetEncodeLayer(256, 256, padding=1),
		)
		self.encode4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			UnetEncodeLayer(256, 512, padding=1),
			UnetEncodeLayer(512, 512, padding=1),
		)
		self.encode5 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			UnetEncodeLayer(512, 1024, padding=1),
			UnetEncodeLayer(1024, 1024, padding=1),
		)
		self.upscale1 = nn.Sequential(
			nn.ConvTranspose2d(1024, 512,kernel_size=2, stride=2)
		)
		self.decode_forward1 = nn.Sequential(
			UnetForwardDecodeLayer(1024,512, padding=1)
		)
		self.upscale2 = nn.Sequential(
			nn.ConvTranspose2d(512, 256,kernel_size=2, stride=2)
		)
		self.decode_forward2 = nn.Sequential(
			UnetForwardDecodeLayer(512, 256, padding=1)
		)
		self.upscale3 = nn.Sequential(
			nn.ConvTranspose2d(256, 128,kernel_size=2, stride=2)
		)
		self.decode_forward3 = nn.Sequential(
			UnetForwardDecodeLayer(256,128,padding=1)
		)
		self.upscale4 = nn.Sequential(
			nn.ConvTranspose2d(128, 64,kernel_size=2, stride=2)
		)
		self.decode_forward4 = nn.Sequential(
			UnetForwardDecodeLayer(128,64, padding=1),
			nn.Conv2d(64, num_classes, kernel_size=1)
		)
	def forward(self, x: torch.Tensor, context=None):
				
		self.x1 = self.encode1(x)
		self.x2 = self.encode2(self.x1)
		self.x3 = self.encode3(self.x2)
		self.x4 = self.encode4(self.x3)
		self.x5 = self.encode5(self.x4)

		y1 = self.upscale1(self.x5)
		c1 = torch.concat((self.x4, y1), 1)
		y2 = self.decode_forward1(c1)
		
		y2 = self.upscale2(y2)
		c2 = torch.concat((self.x3, y2), 1)
		y3 = self.decode_forward2(c2)

		y3 = self.upscale3(y3)
		c3 = torch.concat((functional.center_crop(y3, self.x2.shape[2]), self.x2), 1)
		y4 = self.decode_forward3(c3)

		y4 = self.upscale4(y4)
		c4 = torch.concat((self.x1, y4), 1)
		segmap = self.decode_forward4(c4)
		return segmap
```
As you can see this is a standard Pytorch module. It is contained in nets.py
and it uses modules like ```UnetEncodeLayer``` or ```UnetForwardDecodeLayer```
that can be found in blocks.py.
If you want to write your own model, just remember to define the following class attributes:
+ ```self.output_logits=True```
+ ```self.requires_context=False```
+ ```self.wrapper=False```
if you need more details please refer to the full documentation.
## 2. Register the model
We alredy provide 4 pre-registered models that you can directly initialize from
the configuration file. If you want to register your own model (for example the one that you just wrote) you can do it by modifying the ```load_model``` function contained in utils.py.
Just add at the end of the function
```
elif netname == 'Unet':
	return nets.Unet(classes).to(device)
```
and you are done.
## 3. Write the training config
We implemented several functions to cover most of the needs related to remote sensing image segmentation. The following is just an example, but you can find more information in the full documentation.
```yaml
#-------------DEBUGGING-----------
debug: False
debug_plot: True
verbose: True
#-------------DIRECTORIES------------
train: D:\Datasets\Patches_512x512
validation: D:\Datasets\GID15_urban\GID15\Validation
test: D:\Datasets\GID15_urban\GID15\Test
checkpoint_directory: D:\weights\test\balanced
load_checkpoint: D:\weights\test\test_dino\pretrain_checkpoint20
#--------DEVICE--------------
device: gpu
#----------NET---------------
net: Unetv2
#-------DATA LOADING----------
epochs: 50
chunk_size: 1680
validation_chunk_size: 5
patch_size: 224
batch_size: 10
augmentation: True
# ------LOSS AND OPTIMIZATION----------
loss: CEL
opt: SGD2
ignore_background: True
#-------ALMOST NEVER TOUCH THESE--------
freq: 1
precision_evaluation_freq: 1
num_classes: 15
load_color_mask: False
load_context: False
```