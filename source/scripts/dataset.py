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
import random
import matplotlib.pyplot as plt

class GID15(Dataset):
	'''
	GID15 Dataset.
	This dataset considers crops of the given shape of the original images.
	Cropped pictures are indexed row-major inside an image, and progressively across images based on their directory order.
	
	Parameters:
		root: root directory of the dataset files
		patch_shape: (width, hegith) crop's shape
		transforms: torchvision transforms to apply on images
		target_transforms: torchvision transforms to apply on masks
		color_masks: selects if masks should be index masks (False) or color masks (True)
	'''
	def __init__(self, root, patch_shape = (224, 224), transforms=None, target_transforms=None, color_masks=False, random_shift=False):
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
		self.random_shift = random_shift
		# height, width
		self.image_shape = (6800, 7200)
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
			#self.last_image = tv_tensors.Image(Image.open(filepath))[1:,:,:]
			self.last_image = tv_tensors.Image(Image.open(filepath))
			if self.color_masks:
				#self.last_target = tv_tensors.Image(Image.open(self.__get_color_mask_path(filepath)))[1:,:,:]
				self.last_target = tv_tensors.Image(Image.open(self.__get_color_mask_path(filepath)))
			else:
				self.last_target = tv_tensors.Mask(Image.open(self.__get_idx_mask_path(filepath)))
		# tile index, row major ordering
		tile_idx = idx % self.tiles_per_img
		# row, col position of the tile in our full image
		tile_pos = (tile_idx // self.tiles_in_img_shape[1], tile_idx % self.tiles_in_img_shape[1])
		tile_px_pos = (tile_pos[0] * self.patch_shape[0], tile_pos[1] * self.patch_shape[1])
		tly, tlx = tile_px_pos
		if self.random_shift:
			offset_y = random.randint(-20,20)
			offset_x = random.randint(-20,20)
			tly += offset_y
			tlx += offset_x
		tif_img = v2.functional.crop(self.last_image, tly, tlx, self.patch_shape[0], self.patch_shape[1])
		mask_img = v2.functional.crop(self.last_target, tly, tlx, self.patch_shape[0], self.patch_shape[1])	
		#crop context around patch (automatic padding)
		c_tly = tly-self.patch_shape[0]
		c_tlx = tlx-self.patch_shape[1]
		h = w = self.patch_shape[0]*3
		context = v2.functional.crop(self.last_image, c_tly, c_tlx, h, w)
		# mask should be transofrmed geometrically too!
		if self.transforms:
			tif_img = self.transforms(tif_img)
			context = self.transforms(context)			
		if self.target_transforms:
			mask_img = self.target_transforms(mask_img)
		context = self.resize(context)
		return (tif_img, mask_img, context)
	
	def __get_color_mask_path(self, base_path):
		return os.path.join(self.clrmask_dir, Path(base_path).stem + '_15label.tif')

	def __get_idx_mask_path(self, base_path):
		return os.path.join(self.idxmask_dir, Path(base_path).stem + '_15label.png')

	def get_class_weights(self, classes, distribution=False):
		'''
		Returns class weights over the dataset.
		params
		distribution: if set to True, it will return the priors of the labels instead of the weights.
		'''
		if not self.class_weights:
			self.class_weights = torch.zeros(classes, dtype=torch.float32)
			for img in self.files:
				mask = tv_tensors.Mask(Image.open(self.__get_idx_mask_path(img)))
				for cl in range(classes):
					self.class_weights[cl] += torch.sum(mask == cl)
			if distribution:				
				self.class_weights /= torch.sum(self.class_weights)
			else:
				self.class_weights = torch.sum(self.class_weights) / self.class_weights # we need the inverse: low probability should get big weight

		return self.class_weights
	

class IterableChunk(torch.utils.data.IterableDataset):

    def __init__(self, chunk, images, indexdir, maskdir, random_shift=False):
        super(IterableChunk).__init__()
        self.indexdir = indexdir
        self.maskdir = maskdir
        self.p = 224
        self.tpe = 960 # tiles per image
        self.random_shift = random_shift
        self.image_shape = (6800, 7200)
        self.tiles_in_img_shape = (self.image_shape[0] // self.p, self.image_shape[1] // self.p) # (30,32)
        self.to_load = [images[idx] for idx in chunk]
        self.chunk_size = len(chunk)
        self.chunk_crops = [_ for _ in range(self.tpe*self.chunk_size)] # [0...1919] if chunk size is 2
        random.shuffle(self.chunk_crops)
        self.images, self.index_masks, self.color_masks = self.load_images(self.to_load)
        self.resize = v2.Resize(self.p)
        # shuffle chunk_crops and crop the coresponding image
        # still need to shuffle chunk_crops
        self.patches = [] # list of tuples (image_crop, index_crop, mask_crop, context_crop)
        for x in self.chunk_crops:            
            target_image = x // self.tpe
            tile_idx = x % self.tpe
            tile_pos = (tile_idx // self.tiles_in_img_shape[1], tile_idx % self.tiles_in_img_shape[1])
            tly, tlx = (tile_pos[0] * self.p, tile_pos[1] * self.p)

            # apply random shift
            if self.random_shift:
                offset_y = random.randint(-20,20)
                offset_x = random.randint(-20,20)
                tly += offset_y
                tlx += offset_x

            # get index mask name and color mask name

            # crop image, index mask and color mask
            patch = v2.functional.crop(self.images[target_image], tly, tlx, self.p, self.p)
            index_mask = v2.functional.crop(self.index_masks[target_image], tly, tlx, self.p, self.p)
            color_mask = v2.functional.crop(self.color_masks[target_image], tly, tlx, self.p, self.p)

            #locate, crop and resize context
            c_tly = tly-self.p
            c_tlx = tlx-self.p
            h = w = self.p*3
            context = self.resize(v2.functional.crop(self.images[target_image], c_tly, c_tlx, h, w))            
            # append everything to list
            self.patches.append((patch, index_mask, color_mask, context))
            
    def load_images(self, names):
        """
        Parameters:
            names (list): list of full paths of images to load
        Returns:
            images (list): List of pre-loaded images
            index_masks (list): List of pre-loaded index masks
            color_masks (list): List of pre-loaded color masks
        """
        print("Loading chunk:")
        for name in names:
            print(name)
        images = [tv_tensors.Image(Image.open(name)) for name in names]
        index_masks = [tv_tensors.Mask(Image.open(os.path.join(self.indexdir,Path(name).stem + "_15label.png"))) for name in names]
        color_masks = [tv_tensors.Mask(Image.open(os.path.join(self.maskdir,Path(name).stem + "_15label.tif"))) for name in names]
        return images, index_masks, color_masks

    def __iter__(self):
        return iter(self.patches)
    
    def show_patch(self, index):
        plt.imshow(self.patches[index].permute(1,2,0))
        plt.show()

    
class Loader():
	def __init__(self, root, chunk_size=2, random_shift=False):
		self.root = root
		self.random_shift = random_shift
		self.imdir = os.path.join(root, "Image__8bit_NirRGB")
		self.indexdir = os.path.join(root, "Annotation__index")
		self.maskdir = os.path.join(root, "Annotation__color")
		self.images = sorted([os.path.join(self.imdir, image) for image in os.listdir(self.imdir)])
		self.idxs = [_ for _ in range(len(self.images))]
		self.chunk_size = chunk_size
		self.chunks = None
		assert len(self.images) % self.chunk_size == 0, "Number of images not divisible by chunk size."        
		self.__generate_chunks()

	def shuffle(self):
		random.shuffle(self.idxs)
		self.__generate_chunks()

	def get_iterable_chunk(self,idx):
		"""
		Parameters:
			idx (int): index of chunk that you need to pre-load in memory.
		Returns:
			(IterableChunk): iterator on the specified chunk with shuffled patches.
		"""
		return IterableChunk(self.chunks[idx], self.images, self.indexdir, self.maskdir, self.random_shift)

	def get_chunk(self,idx):
		"""
		Parameters:
			idx (int): index of chunk
		Returns:
			(list): list of names of images belonging to specified chunk            
		"""
		return [self.images[i] for i in self.chunks[idx]]

	def print_chunk(self, idx):
		"""
		Function that prints names of images belonging to chunk
		"""
		for im in self.get_chunk(idx):
			print(im)

	def __generate_chunks(self):
		self.chunks = [[self.idxs[i+(self.chunk_size*offset)] for i in range(self.chunk_size) ] for offset in range (len(self.idxs)//self.chunk_size)]        

	def __len__(self):        
		return len(self.chunks)
