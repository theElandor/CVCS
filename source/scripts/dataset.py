import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import torchvision.transforms as v2
from utils import Converter
from torchvision.transforms.functional import center_crop

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
			if self.augment_mask:
				final_image, final_mask = self.transforms(final_image, final_mask)
			else:
				final_image = self.transforms(final_image)		
		if self.crop:
			final_image = center_crop(final_image, self.crop)
			final_mask = center_crop(final_mask, self.crop)
		return (final_image, final_mask, idx)