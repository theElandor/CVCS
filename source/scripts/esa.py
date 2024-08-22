# Script to make inference on ESA Images

from torchvision import tv_tensors
import torchvision.transforms as transforms
from PIL import Image
import utils
import torch
from converters import GID15Converter
import torchvision.transforms as v2
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import os

def plot(image):
    plt.imshow(image.permute(1,2,0))
    plt.show()


import matplotlib.pyplot as plt
path = "D:\\Datasets\\ESA\\Image__8bit_NirRGB\\t2.png"
image = tv_tensors.Image(Image.open(path))[:3,:,:]
p = 256

_, h, w = image.shape
print(h,w)
plot(image)
c = GID15Converter()

config = {"net":"Unetv2",
          "device":"gpu",
          "num_classes":15,
          "load_checkpoint": "D:\\weights\\checkpoint15"
        }
device = utils.load_device(config)
net = utils.load_network(config, device)
utils.load_checkpoint(config, net)
# with torch.no_grad():
#     net.eval()
#     for i in range(20):
#         patch = rc(image)
#         out = net(patch.unsqueeze(0).type(torch.float32).to(device))
#         pred_index_mask = torch.argmax(out.squeeze().permute(1,2,0).cpu(), dim=2)
#         f, axarr = plt.subplots(1,2)
#         axarr[0].imshow(patch.permute(1,2,0).cpu())
#         axarr[1].imshow(c.iconvert(pred_index_mask))
#         #plt.savefig(os.path.join("output", f"{index}.png"))
#         plt.show()
ht = h // p
wt = w // p
with torch.no_grad():
    net.eval()
    for i in range(ht):
        for j in range(wt):
            patch = v2.functional.crop(image, p*i, p*j, p,p)            
            out = net(patch.unsqueeze(0).type(torch.float32).to(device))
            pred_index_mask = torch.argmax(out.squeeze().permute(1,2,0).cpu(), dim=2)
            tensor = c.iconvert(pred_index_mask)
            arr = ((tensor*255).type(torch.uint8)).numpy()
            im = Image.fromarray(arr)
            index = (i*wt)+j
            im.save(os.path.join("output", f"{index}.png"))

files = os.listdir("output")
sample = ToTensor()(Image.open(os.path.join("output", files[0])))
p = sample.shape[1]

# rows = []
# for r in range(ht):
#     prev = torch.zeros((3,p,p))
#     for h in range(wt):
#         tile = ToTensor()(Image.open(os.path.join("output", f"{(h+r*wt)}.png")))
#         prev = torch.concat((prev, tile), dim=2)
#     rows.append(prev[:,:,p:])
# tot = rows[0]
# for i in range(1, len(rows)):
#     tot = torch.concat((tot, rows[i]), dim=1)    
# save_image(tot, 'output_esa.png')