import torch
import matplotlib.pyplot as plt
import numpy as np

path = "/home/nicola/Desktop/checkpoints/segformer/checkpoint64"
checkpoint = torch.load(path, map_location=torch.device('cpu'))

TL = checkpoint['training_loss_values']
VL = checkpoint['validation_loss_values']



group = 64
means = [np.mean(TL[i:i + group]) for i in range(0, len(TL), group)]
plt.plot(means)

means = [np.mean(VL[i:i + group]) for i in range(0, len(VL), group)]
plt.plot(means)
plt.show()
plt.close()