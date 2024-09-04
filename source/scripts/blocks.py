import torch
import torch.nn as nn
def conv3x3(in_channels: int, out_channels: int, padding=0, dilation=1):
	return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding, dilation=dilation)
def max_pool_2d():
	return nn.MaxPool2d(kernel_size=2, stride=2)

class UnetEncodeLayer(nn.Module):
    # just a standard convolution layer.
	def __init__(self, in_channels: int, out_channels: int, activated=True,max_pool=False, padding=0, dilation=1):
		super(UnetEncodeLayer, self).__init__()
		layers = [			
			conv3x3(in_channels, out_channels, padding=padding, dilation=dilation),
			nn.BatchNorm2d(out_channels),            
		]		
		if activated:
			layers += [nn.ReLU()]
		if max_pool:
			layers += [max_pool_2d()]
		self.layer = nn.Sequential(*layers)
	
	def forward(self, x):
		return self.layer(x)

class UnetUpscaleLayer(nn.Module):
	def __init__(self, scale_factor, in_channels):
		super(UnetUpscaleLayer, self).__init__()
		layers = [
			nn.Upsample(scale_factor = (scale_factor,scale_factor), mode = 'bilinear'),
			conv3x3(in_channels, in_channels//2,padding=1)
		]
		self.layer = nn.Sequential(*layers)
	def forward(self, x):
		return self.layer(x)

class UnetForwardDecodeLayer(nn.Module):
	def __init__(self, in_channels, out_channels, padding=0):
		super(UnetForwardDecodeLayer, self).__init__()
		layers = [
			conv3x3(in_channels=in_channels, out_channels=out_channels, padding=padding),
			nn.ReLU(),
			nn.BatchNorm2d(out_channels),
			conv3x3(in_channels=out_channels, out_channels=out_channels, padding=padding),
			nn.ReLU(),
			nn.BatchNorm2d(out_channels),
		]
		self.layer = nn.Sequential(*layers)
	def forward(self, x):
		return self.layer(x)