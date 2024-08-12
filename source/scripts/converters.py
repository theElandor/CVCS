import torch

class GID15Converter:
	def __init__(self):
		self.color_to_label={
		  (0,    0,    0): 0, # unlabeled
          (200,    0,  0): 1, # industrial land
          (250,    0,150): 2, # urban residential
          (200, 150, 150): 3, # rural residential
          (250, 150, 150): 4, # traffic land
          (0,     200, 0): 5, # paddy field
          (150,  250,  0): 6, # irrigated cropland
          (150, 200, 150): 7, # dry cropland
          (200,   0, 200): 8, # garden plot
          (150,   0, 250): 9, # arbor woodland
          (150,  150,250): 10, # shrub land
          (250,  200,  0): 11, # natural grass land
          (200,  200,  0): 12, # artificial grass land
          (0,     0, 200): 13, # river
          (0,   150, 200): 14, # lake
          (0,   200, 250): 15, # pond
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