import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import torchvision.transforms as v2
from utils import Converter

class PostDamDataset(Dataset):
	def __init__(self, img_dir, masks_dir, extension,transforms=None):
		self.idir = img_dir
		self.mdir = masks_dir		
		self.transforms = transforms		
		self.extension = extension
		self.items = os.listdir(self.idir)
		self.files = [item for item in self.items if os.path.isfile(os.path.join(self.idir, item))]
		self.c = Converter()
	def __len__(self):
		return len(self.files)
	def __getitem__(self, idx):
		img_path = os.path.join(self.idir, "Image_{}{}".format(idx, self.extension))
		mask_path = os.path.join(self.mdir, "Label_{}{}".format(idx, self.extension))
		tif_img = Image.open(img_path)
		tif_mask = Image.open(mask_path)
		if self.transforms:  # if transforms are provided, apply them
			final_image = self.transforms(ToTensor()(tif_img))
		# no transform is applied on mask obv.
		else:
			final_image = ToTensor()(tif_img)
		return (final_image, self.c.convert(ToTensor()(tif_mask)), idx)

transforms = v2.Compose([
    v2.GaussianBlur(kernel_size=(15), sigma=5),
    v2.ElasticTransform(alpha=200.0)
])
