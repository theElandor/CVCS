import yaml
import sys
import converters
import utils
import torch
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from PIL import Image
import os
from dataset import GID15

inFile = sys.argv[1]

with open(inFile,"r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
utils.display_configs(config)

device = utils.load_device(config)

try:
    net = utils.load_network(config, device).to(device)
except:
    print("Error in loading network.")
    exit(0)
TL, VL, mIoU, wIoU = utils.load_checkpoint(config, net)
p = config['patch_size']
dataset = GID15(config['dataset'], (p, p))

if 'range' in config.keys():
    lb, ub = config['range']
    indexes = [_ for _ in range(lb, ub)]
else:    
    indexes = [_ for _ in range(len(dataset))]
c = converters.GID15Converter()
if 'mask_only' in config.keys():
    mask_only = config['mask_only']
else:
    mask_only = False
utils.inference(net, dataset, indexes, device, c, mask_only=mask_only)


if 'out_image' in config.keys():
    files = os.listdir("output")
    sample = ToTensor()(Image.open(os.path.join("output", files[0])))
    p = sample.shape[1]
    Wn = 7200//p
    Hn = 6800//p

    rows = []
    for r in range(Hn):
        prev = torch.zeros((3,p,p))
        for h in range(Wn):
            tile = ToTensor()(Image.open(os.path.join("output", f"{(lb + h+r*Wn)}.png")))
            prev = torch.concat((prev, tile), dim=2)
        rows.append(prev[:,:,p:])
    tot = rows[0]
    for i in range(1, len(rows)):
        tot = torch.concat((tot, rows[i]), dim=1)    
    save_image(tot, config['out_image'])