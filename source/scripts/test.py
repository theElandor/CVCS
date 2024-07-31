import torch
import os
import random
from torchvision import tv_tensors
from PIL import Image
import torchvision.transforms as v2
import matplotlib.pyplot as plt
from pathlib import Path

class IterableChunk(torch.utils.data.IterableDataset):

    def __init__(self, chunk, images, indexdir, maskdir):
        super(IterableChunk).__init__()
        self.indexdir = indexdir
        self.maskdir = maskdir
        self.p = 224
        self.tpe = 960 # tiles per image
        self.random_shift = False
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
        print(max(self.chunk_crops))
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
    def __init__(self, root, chunk_size=2):
        self.root = root
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
        return IterableChunk(self.chunks[idx], self.images, self.indexdir, self.maskdir)
    
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


epochs = 1
test = "D:\\Datasets\\GID15\\gid-15\\GID\\Train"
l = Loader(test, chunk_size = 1)
for epoch in range(epochs):
    l.shuffle()
    for c in range(len(l)):
        dataset = l.get_iterable_chunk(c)
        dl = torch.utils.data.DataLoader(dataset)        
    print("-------------------------------")