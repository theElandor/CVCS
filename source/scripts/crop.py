import torchvision.utils as utils
import os
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision import tv_tensors

def crop_and_save(image, index, p, dest):
    c, h, w = image.shape
    hn = h // p
    wn = w // p
    for i in range(hn): # 0...30
        for j in range(wn): # 0...32
            if c == 4:
            	tile = image[1:, (i*p):(i*p)+p, (j*p):(j*p)+p]
            else:
                tile = image[:, (i*p):(i*p)+p, (j*p):(j*p)+p]
            utils.save_image(tile, os.path.join(dest, f"{(index)*(hn*wn)+(i*wn)+j}.png"))

# NEED TO SPECIFY ROOT DIRECTORY.
# PATCHES WILL BE SAVED IN Patches_224x224/ directory.
p = 224
root = "D:\\Datasets\\GID15\\Train"
total_cropped = 3

folders = ["Annotation__index", "Annotation__color", "Image__8bit_NirRGB"]
dest = "Patches_224x224"
for folder in folders:
    Path(os.path.join(root, dest, folder)).mkdir(parents = True, exist_ok=True)
for folder in folders:
    source = os.path.join(root, folder)
    image_files = [item for item in os.listdir(source)]
    for i,image in enumerate(image_files[:total_cropped]):
        image_path = os.path.join(source, image)        
        im = ToTensor()(Image.open(image_path))        
        crop_and_save(im, i, p,os.path.join(root, dest, folder))