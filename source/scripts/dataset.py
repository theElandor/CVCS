import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision import tv_tensors
import torchvision.transforms as v2
from torchvision.transforms.functional import center_crop
from pathlib import Path
import torch
import converters
from abc import ABC, abstractmethod

class PostDamDataset(Dataset):
	def __init__(self, img_dir, masks_dir, extension,transforms=None, crop=None, augment_mask=False):
		self.idir = img_dir
		self.mdir = masks_dir		
		self.transforms = transforms		
		self.extension = extension
		self.items = os.listdir(self.idir)
		self.files = [item for item in self.items if os.path.isfile(os.path.join(self.idir, item))]
		self.c = converters.PostdamConverter()
		self.crop = crop
		self.augment_mask = augment_mask
	def __len__(self):
		return len(self.files)
	def __getitem__(self, idx):
		img_path = os.path.join(self.idir, "Image_{}{}".format(idx, self.extension))
		mask_path = os.path.join(self.mdir, "Label_{}{}".format(idx, self.extension))
		tif_img = Image.open(img_path)
		tif_mask = Image.open(mask_path)
		final_mask = self.c.convert(ToTensor()(tif_mask))
		final_image = ToTensor()(tif_img)
		if self.transforms:  # if transforms are provided, apply them
			if self.augment_mask:
				final_image, final_mask = self.transforms(final_image, final_mask)
			else:
				final_image = self.transforms(final_image)		
		if self.crop:
			final_image = center_crop(final_image, self.crop)
			final_mask = center_crop(final_mask, self.crop)
		return (final_image, final_mask, idx)

class GF5BP(Dataset, ABC):
	'''
	5 Billion Pixels Dataset.
	This dataset considers crops of the given shape of the original images.
	Cropped pictures are indexed row-major inside an image, and progressively across images based on their directory order.
	
	Parameters:
		root: root directory of the dataset files
		patch_shape: (width, hegith) crop's shape
		transforms: torchvision transforms to apply on images
		target_transforms: torchvision transforms to apply on masks
		color_masks: selects if masks should be index masks (False) or color masks (True)
	'''
	def __init__(self, root, patch_shape = (224, 224), transforms=None, target_transforms=None, color_masks=False):
		self.idir = os.path.join(root, 'Image__8bit_NirRGB')
		self.idxmask_dir = os.path.join(root, 'Annotation__index')
		self.clrmask_dir = os.path.join(root, 'Annotation__color')
		self.color_masks = color_masks
		self.transforms = transforms		
		self.target_transforms = target_transforms
		self.files = [os.path.join(self.idir,item) for item in os.listdir(path=self.idir)]
		self.patch_shape = patch_shape		
		self.last_image = None
		self.last_target = None
		self.last_image_idx = -1		
		# height, width
		self.image_shape = (6908, 7300)
		# rows, cols
		self.tiles_in_img_shape = (self.image_shape[0] // patch_shape[0], self.image_shape[1] // patch_shape[1])
		self.tiles_per_img = self.tiles_in_img_shape[0] * self.tiles_in_img_shape[1]
		self.class_weights = None
		self.resize = v2.Resize(224)
	
	def __len__(self):
		return len(self.files) * self.tiles_per_img
	
	def __getitem__(self, idx):
		base_idx = (self.last_image_idx) * self.tiles_per_img
		# check if image is already loaded
		if (idx >= base_idx + self.tiles_per_img) or (idx < base_idx):
			# load new image
			self.last_image_idx = idx // self.tiles_per_img
			filepath = self.files[self.last_image_idx]
			self.last_image = tv_tensors.Image(Image.open(filepath))[1:,:,:]
			if self.color_masks:
				self.last_target = tv_tensors.Image(Image.open(self.__get_color_mask_path(filepath)))[1:,:,:]
			else:
				self.last_target = tv_tensors.Mask(Image.open(self.__get_idx_mask_path(filepath)))
		# tile index, row major ordering
		tile_idx = idx % self.tiles_per_img
		# row, col position of the tile in our full image
		tile_pos = (tile_idx // self.tiles_in_img_shape[1], tile_idx % self.tiles_in_img_shape[1])
		tile_px_pos = (tile_pos[0] * self.patch_shape[0], tile_pos[1] * self.patch_shape[1])
		tif_img = v2.functional.crop(self.last_image, tile_px_pos[0], tile_px_pos[1], self.patch_shape[0], self.patch_shape[1])
		mask_img = v2.functional.crop(self.last_target, tile_px_pos[0], tile_px_pos[1], self.patch_shape[0], self.patch_shape[1])
		#crop context around patch (automatic padding)
		tly = tile_px_pos[0]-self.patch_shape[0]
		tlx = tile_px_pos[1]-self.patch_shape[1]
		h = w = self.patch_shape[0]*3
		context = v2.functional.crop(self.last_image, tly, tlx, h, w)
		# mask should be transofrmed geometrically too!
		if self.transforms:
			tif_img = self.transforms(tif_img)
			context = self.transforms(context)			
		if self.target_transforms:
			mask_img = self.target_transforms(mask_img)
		context = self.resize(context)
		return (tif_img, mask_img, context)
	
	def __get_color_mask_path(self, base_path):
		return os.path.join(self.clrmask_dir, Path(base_path).stem + '_24label.tif')

	def __get_idx_mask_path(self, base_path):
		return os.path.join(self.idxmask_dir, Path(base_path).stem + '_24label.png')

	def get_class_weights(self):
		'''
		Returns class weights (probabilities) over the dataset.
		'''
		if not self.class_weights:
			self.class_weights = torch.zeros(25, dtype=torch.float32)
			for img in self.files:
				mask = tv_tensors.Mask(Image.open(self.__get_idx_mask_path(img)))
				for cl in range(25):
					self.class_weights[cl] += torch.sum(mask == cl)
			self.class_weights = torch.sum(self.class_weights) / self.class_weights # we need the inverse: low probability should get big weight

		return self.class_weights

class Cropped5BP(Dataset):
	"""
	5BP datasets for cropped images.
	"""
	def __init__(self, root, inference=False):
		self.image_dir = os.path.join(root, 'Image__8bit_NirRGB')
		self.index_dir = os.path.join(root, 'Annotation__index')
		self.color_dir = os.path.join(root, 'Annotation__color')
		self.files = [os.path.join(self.image_dir,item) for item in os.listdir(path=self.image_dir)]		
		self.inference = inference
		self.class_weights = None
	def __len__(self):
		files = [os.path.join(self.image_dir,item) for item in os.listdir(path=self.image_dir)]		
		return len(files)
	def __getitem__(self, idx):
		image = tv_tensors.Image(Image.open(os.path.join(self.image_dir, str(idx)+".png")))
		index = tv_tensors.Mask(Image.open(os.path.join(self.index_dir, str(idx)+".png")))[0,:,:]
		color = ToTensor()(Image.open(os.path.join(self.color_dir, str(idx)+".png")))		
		if self.inference:
			return (image, index, color)
		else: return (image, index) #color

	# NEED REFACTOR, REPLICATED CODE!
	def __get_idx_mask_path(self, base_path):
		return os.path.join(self.index_dir, Path(base_path).stem + '.png')

	def get_class_weights(self):
		'''
		Returns class weights (probabilities) over the dataset.
		'''
		if not self.class_weights:
			self.class_weights = torch.zeros(25, dtype=torch.float32)
			for img in self.files:
				mask = tv_tensors.Mask(Image.open(self.__get_idx_mask_path(img)))
				for cl in range(25):
					self.class_weights[cl] += torch.sum(mask == cl)
			self.class_weights = torch.sum(self.class_weights) / self.class_weights # we need the inverse: low probability should get big weight

		return self.class_weights