import torch

class PostdamConverter: # WARNING: POSTDAM CONVERTER
	def __init__(self):
		self.color_to_label = {
            (1, 1, 0): 0,  # Yellow (cars)
            (0, 1, 0): 1, # Green (trees)
            (0, 0, 1): 2, # Blue (buildings)
            (1, 0, 0): 3,  # Red (clutter)
            (1, 1, 1): 4, # White(impervious surface),
            (0, 1, 1): 5 # Aqua (low vegetation)
        }
	def iconvert(self, mask):
		"""
		Function needed to convert the class label mask needed by CrossEntropy Function
		to the original mask.
		input: class label mask, HxW
		output: original mask, HxWx3
		"""
		H,W = mask.shape
		colors = torch.tensor(list(self.color_to_label.keys())).type(torch.float64)
		labels = torch.tensor(list(self.color_to_label.values())).type(torch.float64)
		output = torch.ones(H,W,3).type(torch.float64)
		for color, label in zip(colors, labels):
			match = (mask == label)
			output[match] = color
		return output
	def convert(self,mask):
		"""
		Function needed to convert the RGB (Nx3x300x300) mask into a 
		'class label mask' needed when computing the loss function.
		In this new representation for each pixel we have a value
		between [0,C) where C is the number of classes, so 6 in this case.
		This new tensor will have shape Nx300x300.
		"""			
		C,H,W = mask.shape
		colors = torch.tensor(list(self.color_to_label.keys()))
		labels = torch.tensor(list(self.color_to_label.values()))
		reshaped_mask = mask.permute(1, 2, 0).reshape(-1, 3)
		class_label_mask = torch.zeros(reshaped_mask.shape[0], dtype=torch.long)
		for color, label in zip(colors, labels):
			match = (reshaped_mask == color.type(torch.float64)).all(dim=1)
			class_label_mask[match] = label
		class_label_mask = class_label_mask.reshape(H,W)		
		return class_label_mask

class GaofenConverter:
	def __init__(self):
		self.color_to_label={
			(0,   0,    0)  : 0, #unlabeled
            (200, 0,    0)  : 1, #industrial area
            (0,  200,   0)  : 2, #paddy field
            (150, 250,  0)  : 3, #irrigated field
            (150, 200,  150): 4, #dry cropland
            (200, 0,    200): 5, #garden land
            (150, 0,    250): 6, #arbor forest
            (150, 150,  250): 7, #shrub forest
            (200, 150,  200): 8, #park
            (250, 200,  0)  : 9, #natural meadow
            (200, 200,  0)  : 10, #artificial meadow
            (0,   0,   200) : 11, #river
            (250, 0,   150) : 12, #urban residential
            (0,   150, 200) : 13, #lake
            (0,   200, 250) : 14, #pond
            (150, 200, 250) : 15, #fish pond
            (250, 250, 250) : 16, #snow
            (200, 200, 200) : 17, #bareland
            (200, 150, 150) : 18, #rural residential
            (250, 200, 150) : 19, #stadium
            (150, 150,   0) : 20, #square
            (250, 150, 150) : 21, #road
            (250, 150,   0) : 22, #overpass
            (250, 200, 250) : 23, #railway station
            (200, 150,   0) : 24, #airport
        }
	def iconvert(self, mask):
		"""
		Function needed to convert the class label mask needed by CrossEntropy Function
		to the original mask.
		input: class label mask, HxW
		output: original mask, HxWx3
		"""
		H,W = mask.shape
		colors = torch.tensor(list(self.color_to_label.keys())).type(torch.float32)/255
		labels = torch.tensor(list(self.color_to_label.values())).type(torch.int)
		output = torch.ones(H,W,3).type(torch.float32)
		for color, label in zip(colors, labels):
			match = (mask == label)
			output[match] = color
		return output
