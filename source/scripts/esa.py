# Script to make inference on ESA Images

from torchvision import tv_tensors
import torchvision.transforms as transforms
from PIL import Image
import utils
import torch
from converters import GID15Converter

def plot(image):
    plt.imshow(image.permute(1,2,0))
    plt.show()


import matplotlib.pyplot as plt
path = "D:\\Datasets\\ESA\\Image__8bit_NirRGB\\modena.png"
image = tv_tensors.Image(Image.open(path))[:3,:,:]
c = GID15Converter()
rc = transforms.RandomCrop((224,224))

config = {"net":"Resnet101",
          "device":"gpu",
          "num_classes":15,
          "load_checkpoint": "D:\\weights\\checkpoint12"
        }
device = utils.load_device(config)
net = utils.load_network(config, device)
utils.load_checkpoint(config, net)
with torch.no_grad():
    net.eval()
    for i in range(20):
        patch = rc(image)
        out = net(patch.unsqueeze(0).type(torch.float32).to(device))
        pred_index_mask = torch.argmax(out.squeeze().permute(1,2,0).cpu(), dim=2)
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(patch.permute(1,2,0).cpu())
        axarr[1].imshow(c.iconvert(pred_index_mask))
        #plt.savefig(os.path.join("output", f"{index}.png"))
        plt.show()