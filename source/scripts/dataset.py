import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision import tv_tensor
import torchvision.transforms as v2
from utils import Converter
from torchvision.transforms.functional import center_crop
from pathlib import Path

class PostDamDataset(Dataset):
	def __init__(self, img_dir, masks_dir, extension,transforms=None, crop=None, augment_mask=False):
		self.idir = img_dir
		self.mdir = masks_dir		
		self.transforms = transforms		
		self.extension = extension
		self.items = os.listdir(self.idir)
		self.files = [item for item in self.items if os.path.isfile(os.path.join(self.idir, item))]
		self.c = Converter()
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
<<<<<<< HEAD
			if self.augment_mask:
				final_image, final_mask = self.transforms(final_image, final_mask)
			else:
				final_image = self.transforms(final_image)		
		if self.crop:
			final_image = center_crop(final_image, self.crop)
			final_mask = center_crop(final_mask, self.crop)
		return (final_image, final_mask, idx)
=======
			final_image = self.transforms(ToTensor()(tif_img))
		# no transform is applied on mask obv.
		else:
			final_image = ToTensor()(tif_img)
		return (final_image, self.c.convert(ToTensor()(tif_mask)), idx)


class GF5BP(Dataset):
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
				self.last_target = tv_tensors.Image(Image.open(os.path.join(self.clrmask_dir, Path(filepath).stem + '_24label.tif')))[1:,:,:]
			else:
				self.last_target = tv_tensors.Mask(Image.open(os.path.join(self.idxmask_dir, Path(filepath).stem + '_24label.png')))
		# tile index, row major ordering
		tile_idx = idx % self.tiles_per_img
		# row, col position of the tile in our full image
		tile_pos = (tile_idx // self.tiles_in_img_shape[1], tile_idx % self.tiles_in_img_shape[1])
		tile_px_pos = (tile_pos[0] * self.patch_shape[0], tile_pos[1] * self.patch_shape[1])
		tif_img = v2.functional.crop(self.last_image, tile_px_pos[0], tile_px_pos[1], self.patch_shape[0], self.patch_shape[1])
		mask_img = v2.functional.crop(self.last_target, tile_px_pos[0], tile_px_pos[1], self.patch_shape[0], self.patch_shape[1])
		# mask should be transofrmed geometrically too!
		if self.transforms:
			tif_img = self.transforms(tif_img)
		if self.target_transforms:
			mask_img = self.target_transforms(mask_img)

		return (tif_img, mask_img)
			

>>>>>>> 9e7c41ec382e0ce3053cfee50e16da009bad041c
