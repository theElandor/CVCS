from blocks import UnetEncodeLayer,UnetUpscaleLayer,UnetForwardDecodeLayer, conv3x3
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as functional
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from blocks import UnetEncodeLayer,UnetUpscaleLayer,UnetForwardDecodeLayer
from torchvision.models.segmentation import deeplabv3_resnet101,deeplabv3_mobilenet_v3_large,deeplabv3_resnet50
from torch.nn import ConvTranspose2d, Conv2d

"""
To write a new network make sure that your class is initialized
with the following attributes to make it trainable
and compatible with the codebase:

+ self.requires_context <bool>:
	set this to True if the network needs the context around the input
	for the forward pass. This way the training script will move 
	the context to gpu only if needed.

+ self.wrapper <bool>:
	If set to true, the script that loads a checkpoint will call the
	self.custom_load() method instead of trying to directly load weights.
	The self.custom_load() method must load the model in the self.model
	parameter, and the class just serves as a wrapper.
	Look at DeepLabv3Resnet101 as an example.

+ self.returns_logits <bool>:
	Set this to true if the model performs argmax on the logits inside
	the forward pass. Usually it's not the case, so you can set this to 
	False in most networks.
"""
class Urnet(nn.Module):
      # classic Unet with some reshape and cropping to match our needs.
	def __init__(self, num_classes):
		super(Urnet, self).__init__()		
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
	def forward(self, x: torch.Tensor, context:torch.Tensor=None):
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

class Urnetv2(nn.Module):
      # classic Unet with some reshape and cropping to match our needs.
	def __init__(self, num_classes):
		super(Urnetv2, self).__init__()		
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

def aux_hook(module, input, output):
    module.aux_out = output

class google_backbone(torch.nn.Module):
    def __init__(self):
        super(google_backbone, self).__init__()
        googleNet = torchvision.models.googlenet(weights='DEFAULT')
        self.backbone = torch.nn.Sequential(*list(googleNet.children())[:-6])
        self.out_layer = torch.nn.Conv2d(832, 960, kernel_size=(1, 1))
        self.aux_layer = torch.nn.Conv2d(192, 40, kernel_size=(1, 1))
        self.backbone[4].register_forward_hook(aux_hook)

    def forward(self, x):
        x = self.backbone(x)
        return {'out': self.out_layer(x), 'aux': self.aux_layer(self.backbone[4].aux_out)}

class resnet18_backbone(torch.nn.Module):
    def __init__(self):
        super(resnet18_backbone, self).__init__()
        self.backbone = torchvision.models.resnet18(weights='DEFAULT')
        self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-2])
        self.out_layer = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 960, kernel_size=(1, 1))
        )
        self.aux_layer = torch.nn.Conv2d(128, 40, kernel_size=(1, 1))

        self.backbone[5].register_forward_hook(aux_hook)

    def forward(self, x):
        return {'out': self.out_layer(self.backbone(x)), 'aux': self.aux_layer(self.backbone[5].aux_out)}
    
class DeepLabv3Resnet101(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DeepLabv3Resnet101, self).__init__()
        self.requires_context = False
        self.wrapper = True
        self.returns_logits = True
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
        self.load_state_dict(checkpoint_state_dict_mod)

class DeepLabv3Resnet50(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DeepLabv3Resnet50, self).__init__()
        self.requires_context = False
        self.wrapper = True
        self.returns_logits = True
        self.num_classes = num_classes
        if pretrained:
            self.model = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
            in_channels = self.model.classifier[4].in_channels
            self.model.classifier[4] = torch.nn.Conv2d(in_channels, self.num_classes, kernel_size=1)
        else:
            self.model = deeplabv3_resnet101(num_classes=self.num_classes)

    def forward(self, x: torch.Tensor, context=None):
        d = self.model(x)
        return d['out']
    
class DeepLabV3MobileNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, _googlenet_backbone=False, _resnet18_backbone=False):
        super(DeepLabV3MobileNet, self).__init__()
        self.googlenet_backbone = _googlenet_backbone
        self.resnet18_backbone = _resnet18_backbone
        self.requires_context = False
        self.wrapper = True
        self.returns_logits = True

        self.num_classes = num_classes
        if pretrained:
            self.model = deeplabv3_mobilenet_v3_large(
                weights=torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
            in_channels = self.model.classifier[4].in_channels
            self.model.classifier[4] = torch.nn.Conv2d(in_channels, self.num_classes, kernel_size=1)
        else:
            self.model = deeplabv3_mobilenet_v3_large(self.num_classes)

        if self.googlenet_backbone:
            self.model.backbone = google_backbone()
        elif self.resnet18_backbone:
            self.model.backbone = resnet18_backbone()

    def forward(self, x: torch.Tensor, context=None):
        if x.dtype is not torch.float:
            x = x.to(torch.float)
        d = self.model(x)
        return d['out']

    def custom_load(self, checkpoint):
        checkpoint_state_dict_mod = {}
        checkpoint_state_dict = checkpoint['model_state_dict']
        for item in checkpoint_state_dict:
            checkpoint_state_dict_mod[str(item).replace('module', 'model')] = checkpoint_state_dict[item]
        self.load_state_dict(checkpoint_state_dict_mod)

class SegformerMod(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(SegformerMod, self).__init__()
        self.requires_context = False
        self.wrapper = True
        self.returns_logits = True

        self.num_classes = num_classes
        # load pretrained
        if pretrained:
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b3-finetuned-ade-512-512")
        else:
            self.segformer = SegformerForSemanticSegmentation(SegformerConfig())

        # change decoder head to output num_classes channels
        mlp_in_channels = self.segformer.decode_head.classifier.in_channels
        self.segformer.decode_head.classifier = nn.Conv2d(mlp_in_channels, num_classes, kernel_size=(1, 1),
                                                          stride=(1, 1))

        self.seq = torch.nn.Sequential(ConvTranspose2d(num_classes, num_classes, 8, stride=2, padding=3),
                                       torch.nn.ReLU(),
                                       ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1),
                                       torch.nn.ReLU(),
                                       Conv2d(num_classes, num_classes, kernel_size=3, padding=1))

        self.preprocessor = v2.Compose([
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.upsampler = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x: torch.Tensor, context=None):
        x = self.preprocessor(x)
        out = self.segformer(x).logits
        return self.seq(out)

    def custom_load(self, checkpoint):
        checkpoint_state_dict_mod = {}
        checkpoint_state_dict = checkpoint['model_state_dict']
        for item in checkpoint_state_dict:
            checkpoint_state_dict_mod[str(item).replace('module.', '')] = checkpoint_state_dict[item]
        self.load_state_dict(checkpoint_state_dict_mod)