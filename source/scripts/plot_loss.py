# plotting training loss of some checkpoints
import torch
path = "/homes/mlugli/checkpoints/test9/checkpoint18"
checkpoint = torch.load(path, map_location=torch.device('cpu'))
TL = checkpoint['training_loss_values']
VL = checkpoint['validation_loss_values']
import matplotlib.pyplot as plt
import numpy as np
group = 100
means = [np.mean(VL[i:i+group]) for i in range(0, len(VL), group)]
plt.plot(means)
plt.savefig('test.png')
