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
	
class PositionalEncoding(nn.Module):
	# I leave it here but it's not needed if we use convolutional tokenizer.
	def __init__(self, D, num_patches):
		super(PositionalEncoding, self).__init__()
		self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, D))
	def forward(self, x):
		return x + self.pos_embedding
	
class VisionTransformer(nn.Module):
	# D = embedding dimension (patch is p*p*3 and will be projected to be D dimensional)
	# N = number of patches
	# p = patch size	
	def __init__(self, D, num_heads):
		super(VisionTransformer, self).__init__()
		# self.linear_projection = nn.Linear(p*p*3, D) don't need it in this architecture
		# self.positional_encoding = PositionalEncoding(D, N) neither this.
		self.layer_norm1_1 = nn.LayerNorm(D) #q
		self.layer_norm1_2 = nn.LayerNorm(D) #k,v
		self.layer_norm2 = nn.LayerNorm(D)
		self.MHA = nn.MultiheadAttention(embed_dim=D, num_heads=num_heads, batch_first=True)
		self.mlp = nn.Sequential(
			# using D*4 hidden size according to original vision transformer paper
			nn.Linear(D, D*4),
			nn.GELU(),
			nn.Linear(D*4, D)
		)
		# we should have one of this for each head
	def forward(self, s: torch.tensor):
		"""
		Parameters:
			+ s (tensor): concatenation of query, key and value
		"""
		#x = self.linear_projection(x) # N, p*p*3 --> N, D
		#self.r1 = self.positional_encoding(x) # add positional encoding to x, embedded patches
		q = s[0]
		k = s[1]
		v = s[2]
		self.r1 = q
		q = self.layer_norm1_1(q)
		k = self.layer_norm1_2(k)
		v = self.layer_norm1_2(v)

		x = self.MHA(q,k,v)[0]

		self.r2 = x + self.r1
		x = self.layer_norm2(x)
		x = self.mlp(x)
		return torch.stack((x+self.r2, k, v), dim=0)

def vision_transformer(D,num_heads):
	return VisionTransformer(D,num_heads)
	
class VisionTransformerEncoder(nn.Module):
	def __init__(self, D, num_heads, layers):
		super(VisionTransformerEncoder, self).__init__()
		self.layers =[vision_transformer(D,num_heads) for _ in range(layers)]
		self.stack = nn.Sequential(*self.layers)
	def forward(self, q,k,v):
		s = torch.stack((q,k,v), dim=0)
		return self.stack(s)