import torch
import torch.nn as nn
def conv3x3(in_channels: int, out_channels: int, padding=0):
	return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding)
def max_pool_2d():
	return nn.MaxPool2d(kernel_size=2, stride=2)

class UnetEncodeLayer(nn.Module):
    # just a standard convolution layer.
	def __init__(self, in_channels: int, out_channels: int, activated=True,max_pool=False, padding=0):
		super(UnetEncodeLayer, self).__init__()
		layers = [
            conv3x3(in_channels, out_channels, padding=padding),
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
		self.layer_norm1 = nn.LayerNorm(D)
		self.layer_norm2 = nn.LayerNorm(D)
		self.MHA = nn.MultiheadAttention(embed_dim=D, num_heads=num_heads, batch_first=True)
		self.mlp = nn.Sequential(
			# using D*4 hidden size according to original vision transformer paper
			nn.Linear(D, D*4),
            nn.GELU(),
            nn.Linear(D*4, D)
		)
		# we should have one of this for each head
	def forward(self, x):
		#x = self.linear_projection(x) # N, p*p*3 --> N, D
		#self.r1 = self.positional_encoding(x) # add positional encoding to x, embedded patches
		self.r1 = x		
		x = self.layer_norm1(self.r1)		
		x = self.MHA(x,x,x)[0]
		self.r2 = x + self.r1
		x = self.layer_norm2(self.r2)
		x = self.mlp(x)
		return x + self.r2

def vision_transformer(D,num_heads):
    return VisionTransformer(D,num_heads)
    
class VisionTransformerEncoder(nn.Module):
    def __init__(self, D, num_heads, layers):
        super(VisionTransformerEncoder, self).__init__()
        self.layers =[vision_transformer(D,num_heads) for _ in range(layers)]
        self.stack = nn.Sequential(*self.layers)
    def forward(self, x):
        return self.stack(x)