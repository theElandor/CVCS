import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import torchvision.utils as utils
import os

patch_size = 300
for s in ["Image", "Label"]:
    imdir = "/work/cvcs2024/MSseg/Postdam_6000x6000/{}s/".format(s)
    items = os.listdir(imdir)
    files = [item for item in items if os.path.isfile(os.path.join(imdir, item))]
    #output = torch.ones((num_patches, c, patch_size, patch_size))
    for image_index, file in enumerate(files):
        image = ToTensor()(Image.open(os.path.join(imdir, file)))
        c,H,W = image.shape
        assert H % patch_size == 0, "Height not divisibile by patch size."
        oH = H // patch_size
        oW = W // patch_size
        num_patches = oH*oW # 400
        for i in range(oH):
            for j in range(oW):
                #output[(i*oW)+j] = image[:, i*300:(i+1)*patch_size, j*300:(j+1)*patch_size]
                utils.save_image(image[:, i*300:(i+1)*patch_size, j*300:(j+1)*patch_size], "/work/cvcs2024/MSseg/Postdam_6000x6000/{}s/{}_{}.png".format(s,s,(i*oW)+j+image_index*num_patches))
    # num_patches, C, 300,300