import yaml
import sys
from torchvision.utils import save_image
from PIL import Image
import os
import utils
import matplotlib.pyplot as plt
inFile = sys.argv[1]

with open(inFile,"r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print("LOADED CONFIGURATIONS:")
print(config)

TL, VL, mIoU, wIoU = utils.load_checkpoint(config)
plt.plot(wIoU)
plt.show()