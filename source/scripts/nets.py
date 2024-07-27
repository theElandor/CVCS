from blocks import UnetEncodeLayer,UnetUpscaleLayer,UnetForwardDecodeLayer, VisionTransformerEncoder
import torch
import torch.nn as nn
import torchvision.transforms.functional as functional
import math
from transformers import AutoModel, AutoImageProcessor
from blocks import UnetEncodeLayer,UnetUpscaleLayer,UnetForwardDecodeLayer, VisionTransformerEncoder
import torch
import torch.nn as nn
import torchvision.transforms.functional as functional
import math
from transformers import AutoModel
class Urnet(nn.Module):
      # classic Unet with some reshape and cropping to match our needs.
	def __init__(self, num_classes):
		super(Urnet, self).__init__()
		self.residuals = []
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
	def forward(self, x: torch.Tensor):
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

class Tunet(nn.Module): # Unet + vision transformer
    def __init__(self, d, heads, layers):
        super(Tunet, self).__init__()
        self.d = d
        self.heads = heads
        self.layers = layers
        self.residuals = []        
        # encoding part of the Unet vanilla architecture
        self.encode1 = nn.Sequential(
            UnetEncodeLayer(3, d//16, padding=1),
            UnetEncodeLayer(d//16, d//16, padding=1),
        )
        self.encode2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            UnetEncodeLayer(d//16, d//8, padding=1), 
            UnetEncodeLayer(d//8, d//8, padding=1),                        
        )
        self.encode3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            UnetEncodeLayer(d//8, d//4, padding=1),
            UnetEncodeLayer(d//4, d//4, padding=1),
        )
        self.encode4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            UnetEncodeLayer(d//4, d//2, padding=1),
            UnetEncodeLayer(d//2, d//2, padding=1),
        )
        self.encode5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),            
            UnetEncodeLayer(d//2, d, padding=1),
            UnetEncodeLayer(d, d, padding=1),
        )
        self.transformer = VisionTransformerEncoder(self.d,self.heads,self.layers)
        self.upscale1 = nn.Sequential(
            UnetUpscaleLayer(2, d)
        )
        self.decode_forward1 = nn.Sequential(
            UnetForwardDecodeLayer(d,d//2, padding=1)
        )
        self.upscale2 = nn.Sequential(
            UnetUpscaleLayer(2, d//2)
        )
        self.decode_forward2 = nn.Sequential(
            UnetForwardDecodeLayer(d//2, d//4, padding=1)
        )
        self.upscale3 = nn.Sequential(
            UnetUpscaleLayer(2,d//4)
        )
        self.decode_forward3 = nn.Sequential(
            UnetForwardDecodeLayer(d//4,d//8,padding=1)
        )
        self.upscale4 = nn.Sequential(
            UnetUpscaleLayer(2,d//8)
        )
        self.decode_forward4 = nn.Sequential(
            UnetForwardDecodeLayer(d//8,d//16, padding=1),
            nn.Conv2d(d//16, 6, kernel_size=1) # final conv 1x1
            # Model output is 6xHxW, so we have a prob. distribution
            # for each pixel (each pixel has a logit for each of the 6 classes.)
        )
    def forward(self,x):
        self.x1 = self.encode1(x)
        self.x2 = self.encode2(self.x1)
        self.x3 = self.encode3(self.x2)
        self.x4 = self.encode4(self.x3)       
        self.x5 = self.encode5(self.x4)        
        _, _,h,w = self.x5.shape
        self.sequence = self.x5.reshape(-1,h*w, self.d)
        attention_encoded = self.transformer(self.sequence)
        attention_encoded = attention_encoded.reshape(-1,self.d,h,w)
        y1 = self.upscale1(attention_encoded)
        c1 = torch.concat((self.x4, y1), 1)
        y2 = self.decode_forward1(c1)

        y2 = self.upscale2(y2)
        c2 = torch.concat((self.x3, y2), 1)
        y3 = self.decode_forward2(c2)

        y3 = self.upscale3(y3)
        c3 = torch.concat((functional.center_crop(y3, 150), self.x2), 1)
        y4 = self.decode_forward3(c3)

        y4 = self.upscale4(y4)
        c4 = torch.concat((self.x1, y4), 1)
        segmap = self.decode_forward4(c4)
        return segmap

class Swin(nn.Module): # swinT + unet head
    def __init__(self, embed_dim, size, num_classes, device):
        super(Swin, self).__init__()
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
    def forward(self, x):
        inputs = self.image_processor(x, return_tensors="pt")
        self.r1, self.r2, self.r3, _, self.r4 = self.swin(inputs['pixel_values'].to(self.device),return_dict=True, output_hidden_states=True)['hidden_states']
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