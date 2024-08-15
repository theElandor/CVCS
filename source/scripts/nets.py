from blocks import UnetEncodeLayer,UnetUpscaleLayer,UnetForwardDecodeLayer, conv3x3
import torch
import torch.nn as nn
import torchvision.transforms.functional as functional
import math
from transformers import AutoModel, AutoImageProcessor
from blocks import UnetEncodeLayer,UnetUpscaleLayer,UnetForwardDecodeLayer
import torch
import torch.nn as nn
import torchvision.transforms.functional as functional
import math
from transformers import AutoModel
from torchvision.models.segmentation import deeplabv3_resnet101,deeplabv3_mobilenet_v3_large
import torchvision

class Urnet(nn.Module):
      # classic Unet with some reshape and cropping to match our needs.
	def __init__(self, num_classes):
		super(Urnet, self).__init__()		
		self.requires_context = False
		self.wrapper = False
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
			UnetUpscaleLayer(2, 1024)
		)
		self.decode_forward1 = nn.Sequential(
			UnetForwardDecodeLayer(1024,512, padding=1)
		)
		self.upscale2 = nn.Sequential(
			UnetUpscaleLayer(2, 512)
		)
		self.decode_forward2 = nn.Sequential(
			UnetForwardDecodeLayer(512, 256, padding=1)
		)
		self.upscale3 = nn.Sequential(
			UnetUpscaleLayer(2,256)
		)
		self.decode_forward3 = nn.Sequential(
			UnetForwardDecodeLayer(256,128,padding=1)
		)
		self.upscale4 = nn.Sequential(
			UnetUpscaleLayer(2,128)
		)
		self.decode_forward4 = nn.Sequential(
			UnetForwardDecodeLayer(128,64, padding=1),
			nn.Conv2d(64, num_classes, kernel_size=1) # final conv 1x1
			# Model output is 6xHxW, so we have a prob. distribution
			# for each pixel (each pixel has a logit for each of the 6 classes.)
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

class Swin(nn.Module): # swinT + unet head
	def __init__(self, embed_dim, size, num_classes, device):
		super(Swin, self).__init__()	
		self.requires_context = False
		self.wrapper = False
		self.c = embed_dim
		self.h = size
		self.w = size		
		self.num_classes = num_classes        
		self.device = device
		if embed_dim == 96:             
			model_name = "microsoft/swin-tiny-patch4-window7-224"
		elif embed_dim == 128:
			model_name = "microsoft/swin-base-patch4-window7-224"
		self.swin = AutoModel.from_pretrained(model_name)
		self.image_processor = AutoImageProcessor.from_pretrained(model_name)
		self.upscale1 = UnetUpscaleLayer(2, self.c*8)
		self.decode_forward1 = UnetForwardDecodeLayer(self.c*8,self.c*4, padding=1)
		self.upscale2 = UnetUpscaleLayer(2, self.c*4)    
		self.decode_forward2 = UnetForwardDecodeLayer(self.c*4, self.c*2, padding=1)
		self.upscale3 = UnetUpscaleLayer(2,self.c*2)
		self.decode_forward3 = UnetForwardDecodeLayer(self.c*2, self.c, padding=1)
		self.upscale4 = UnetUpscaleLayer(2,self.c)
		self.decode_forward4 = nn.Sequential(
			UnetForwardDecodeLayer(self.c//2,self.c//2,padding=1),
			UnetUpscaleLayer(2, self.c//2),
			nn.Conv2d(self.c//4, num_classes, kernel_size=1) # final conv 1x1
			# Model output is 6xHxW, so we have a prob. distribution
			# for each pixel (each pixel has a logit for each of the 6 classes.)
		)
	def forward(self, x: torch.Tensor, context=None):
		inputs = self.image_processor(x, return_tensors="pt")
		self.r1, self.r2,
		self.r3, _, self.r4 = self.swin(inputs['pixel_values'].to(self.device),return_dict=True, output_hidden_states=True)['hidden_states']
		s = int(math.sqrt(self.r4.shape[1]))
		self.r4 = self.r4.swapaxes(1,2).reshape(-1,self.c*8,s,s)
		self.r3 = self.r3.swapaxes(1,2).reshape(-1,self.c*4,s*2,s*2)
		self.r2 = self.r2.swapaxes(1,2).reshape(-1,self.c*2,s*4,s*4)
		self.r1 = self.r1.swapaxes(1,2).reshape(-1,self.c,s*8,s*8)
		
		x = self.upscale1(self.r4)
		c1 = torch.concat((x, self.r3), 1)
		x = self.decode_forward1(c1)
		x = self.upscale2(x)
		c2 = torch.concat((x, self.r2), 1)
		x = self.decode_forward2(c2)        
		x = self.upscale3(x)
		c3 = torch.concat((x, self.r1), 1)        
		x = self.decode_forward3(c3)
		x = self.upscale4(x)
		x = self.decode_forward4(x)
		return x

class Fusion(nn.Module): # STILL TESTING
	def __init__(self, num_classes, device):
		super(Fusion, self).__init__()
		self.requires_context = True
		self.wrapper = False
		self.device = device		
		model_name = "microsoft/swin-base-patch4-window7-224"
		self.swin = AutoModel.from_pretrained(model_name)
		self.image_processor = AutoImageProcessor.from_pretrained(model_name)		
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
		self.encode6 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			UnetEncodeLayer(1024, 2048, padding=1),
			UnetEncodeLayer(2048, 2048, padding=1),
		)
		self.encode7 = UnetForwardDecodeLayer(2048, 1024, padding=1)
		self.upscale = nn.Sequential(
			nn.Upsample(scale_factor = (2,2), mode = 'bilinear'),
			conv3x3(1024,1024,padding=1),
			nn.BatchNorm2d(1024),
		)
		self.upscale1 = nn.Sequential(
			UnetUpscaleLayer(2, 1024)
		)
		self.decode_forward1 = nn.Sequential(
			UnetForwardDecodeLayer(1024,512, padding=1)
		)
		self.upscale2 = nn.Sequential(
			UnetUpscaleLayer(2, 512)
		)
		self.decode_forward2 = nn.Sequential(
			UnetForwardDecodeLayer(512, 256, padding=1)
		)
		self.upscale3 = nn.Sequential(
			UnetUpscaleLayer(2,256)
		)
		self.decode_forward3 = nn.Sequential(
			UnetForwardDecodeLayer(256,128,padding=1)
		)
		self.upscale4 = nn.Sequential(
			UnetUpscaleLayer(2,128)
		)
		self.decode_forward4 = nn.Sequential(
			UnetForwardDecodeLayer(128,64, padding=1),
			nn.Conv2d(64, num_classes, kernel_size=1) # final conv 1x1
			# Model output is 6xHxW, so we have a prob. distribution
			# for each pixel (each pixel has a logit for each of the 6 classes.)
		)
	def forward(self, x, context):
		self.x1 = self.encode1(x)
		self.x2 = self.encode2(self.x1)
		self.x3 = self.encode3(self.x2)
		self.x4 = self.encode4(self.x3)
		self.x5 = self.encode5(self.x4)
		self.x6 = self.encode6(self.x5)
		self.spatial_features = self.encode7(self.x6)

		inputs = self.image_processor(context, return_tensors="pt")
		r1, r2, r3, _, r4 = self.swin(inputs['pixel_values'].to(self.device),return_dict=True, output_hidden_states=True)['hidden_states']
		
		_,c,h,w = self.spatial_features.shape
		r4 = r4.permute(0,2,1).reshape(-1, c,h,w)
		self.embedding = self.spatial_features + r4
		self.embedding = self.upscale(self.embedding)
		
		y1 = self.upscale1(self.embedding)
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

class UnetTorch(nn.Module):
	def __init__(self, device, in_channels=3, out_channels=16, init_features=224, pretrained=False):
		super(UnetTorch, self).__init__()		
		self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=in_channels, out_channels=out_channels, init_features=init_features, pretrained=pretrained).to(device)
		self.wrapper = True
		self.requires_context = False
	def forward(self, x: torch.Tensor, context=None):
		return self.model(x)

class Urnetv2(nn.Module):
      # classic Unet with some reshape and cropping to match our needs.
	def __init__(self, num_classes):
		super(Urnetv2, self).__init__()		
		self.requires_context = False
		self.wrapper = False
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
			nn.Conv2d(64, num_classes, kernel_size=1) # final conv 1x1
			# Model output is 6xHxW, so we have a prob. distribution
			# for each pixel (each pixel has a logit for each of the 6 classes.)
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
	
class FUnet(nn.Module):
      # classic Unet with some reshape and cropping to match our needs.
	def __init__(self, num_classes):
		super(FUnet, self).__init__()		
		self.requires_context = True
		self.wrapper = False
		# -----------------PATCH ENCODER-----------------------
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
		# -----------------CONTEXT ENCODER--------------------
		self.encode1_c = nn.Sequential(
			UnetEncodeLayer(3, 64, padding=1),
			UnetEncodeLayer(64, 64, padding=1), ## keep dimensions unchanged
		)
		self.encode2_c = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			UnetEncodeLayer(64, 128, padding=1),
			UnetEncodeLayer(128, 128, padding=1),
		)
		self.encode3_c = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			UnetEncodeLayer(128, 256, padding=1),
			UnetEncodeLayer(256, 256, padding=1),
		)
		self.encode4_c = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			UnetEncodeLayer(256, 512, padding=1),
			UnetEncodeLayer(512, 512, padding=1),
		)
		self.encode5_c = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			UnetEncodeLayer(512, 1024, padding=1),
			UnetEncodeLayer(1024, 1024, padding=1),
		)		

		# ---------------DECODER-----------------

		self.upscale1 = nn.Sequential(
			nn.ConvTranspose2d(1024, 512,kernel_size=2, stride=2)
		)
		self.decode_forward1 = nn.Sequential(
			UnetForwardDecodeLayer(1536,512, padding=1)
		)
		self.upscale2 = nn.Sequential(
			nn.ConvTranspose2d(512, 256,kernel_size=2, stride=2)
		)
		self.decode_forward2 = nn.Sequential(
			UnetForwardDecodeLayer(768, 256, padding=1)
		)
		self.upscale3 = nn.Sequential(
			nn.ConvTranspose2d(256, 128,kernel_size=2, stride=2)
		)
		self.decode_forward3 = nn.Sequential(
			UnetForwardDecodeLayer(384,128,padding=1)
		)
		self.upscale4 = nn.Sequential(
			nn.ConvTranspose2d(128, 64,kernel_size=2, stride=2)
		)
		self.decode_forward4 = nn.Sequential(
			UnetForwardDecodeLayer(192,64, padding=1),
			nn.Conv2d(64, num_classes, kernel_size=1) # final conv 1x1
			# Model output is 6xHxW, so we have a prob. distribution
			# for each pixel (each pixel has a logit for each of the 6 classes.)
		)

		self.fusion = nn.Sequential(
			UnetForwardDecodeLayer(2048,1024, padding=1),			
		)

	def encode_patch(self, x: torch.Tensor):
		self.x1 = self.encode1(x)
		self.x2 = self.encode2(self.x1)
		self.x3 = self.encode3(self.x2)
		self.x4 = self.encode4(self.x3)
		self.x5 = self.encode5(self.x4)
		return self.x5

	def encode_context(self, x:torch.Tensor):
		self.x1_c = self.encode1_c(x)
		self.x2_c = self.encode2_c(self.x1_c)
		self.x3_c = self.encode3_c(self.x2_c)
		self.x4_c = self.encode4_c(self.x3_c)
		self.x5_c = self.encode5_c(self.x4_c)
		return self.x5_c
	
	def embedding_fusion(self):
		self.concat_embeddings = torch.concat((self.x5, self.x5_c), 1)		
		self.fused_features = self.fusion(self.concat_embeddings)		

	def decode(self):
		y1 = self.upscale1(self.fused_features)

		c1 = torch.concat((self.x4, self.x4_c, y1), 1)
		y2 = self.decode_forward1(c1)
		
		y2 = self.upscale2(y2)
		c2 = torch.concat((self.x3, self.x3_c, y2), 1)
		y3 = self.decode_forward2(c2)

		y3 = self.upscale3(y3)
		c3 = torch.concat((self.x2, self.x2_c, functional.center_crop(y3, self.x2.shape[2])), 1)
		y4 = self.decode_forward3(c3)

		y4 = self.upscale4(y4)
		c4 = torch.concat((self.x1, self.x1_c, y4), 1)
		return self.decode_forward4(c4)
		
	def forward(self, x: torch.Tensor, context):
		patch_embedding = self.encode_patch(x)
		context_embedding = self.encode_context(context)
		self.embedding_fusion()
		segmap = self.decode()
		return segmap

class DeepLabv3Resnet101(nn.Module):
	def __init__(self, num_classes, pretrained=True):
		super(DeepLabv3Resnet101, self).__init__()		
		self.requires_context = False
		self.wrapper = True
		self.num_classes = num_classes
		if pretrained:
			self.model = deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1')
			in_channels = self.model.classifier[4].in_channels
			self.model.classifier[4] = torch.nn.Conv2d(in_channels, self.num_classes, kernel_size=1)
		else:
			self.model = deeplabv3_resnet101(num_classes=self.num_classes)
	def forward(self, x: torch.Tensor, context=None):
		d = self.model(x)
		return d['out']
	def custom_load(self, checkpoint):
		checkpoint_state_dict_mod = {}
		checkpoint_state_dict = checkpoint['model_state_dict']
		for item in checkpoint_state_dict:
			checkpoint_state_dict_mod[str(item).replace('module.', '')] = checkpoint_state_dict[item]
		self.model.load_state_dict(checkpoint_state_dict_mod)

class DeepLabV3MobileNet(nn.Module):
	def __init__(self, num_classes, pretrained=True):
		super(DeepLabV3MobileNet, self).__init__()		
		self.requires_context = False
		self.wrapper = True
		self.num_classes = num_classes
		if pretrained:
			self.model = deeplabv3_mobilenet_v3_large( weights=torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
			in_channels = self.model.classifier[4].in_channels
			self.model.classifier[4] = torch.nn.Conv2d(in_channels, self.num_classes, kernel_size=1)
		else:
			self.model = deeplabv3_mobilenet_v3_large(self.num_classes)
	def forward(self, x: torch.Tensor, context=None):
		d = self.model(x)
		return d['out']
	def custom_load(self, checkpoint):
		checkpoint_state_dict_mod = {}
		checkpoint_state_dict = checkpoint['model_state_dict']
		for item in checkpoint_state_dict:
			checkpoint_state_dict_mod[str(item).replace('module.', '')] = checkpoint_state_dict[item]
		self.model.load_state_dict(checkpoint_state_dict_mod)	