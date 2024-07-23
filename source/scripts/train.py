import torch
import torch.nn as nn
import numpy as np
from dataset import GF5BP
from nets import Swin,Urnet
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
from tqdm import tqdm
from utils import validation_loss, save_model, eval_model
import matplotlib.pyplot as plt
import os

# START Control variables---------------------------------
# This section contains some variables that need to be set before running the script.
train = "/work/cvcs2024/MSseg/5bp/Train"
validation = "/work/cvcs2024/MSseg/5bp/Validation"
test = "/work/cvcs2024/MSseg/5bp/Test"
checkpoint_directory = "/homes/mlugli/checkpoints/swin1" # directory to save checkpoints in
extension = ".png" # extension of output files if produced.
epochs = 10
load_checkpoint = ""
freq = 1 # checkpoint saving frequency. If set to 2, it will save a checkpoint every 2 epochs.
verbose = False
batch_size = 16 # this parameter will be overwritten if a checkpoint is loaded
precision_evaluation_freq = 2
# checkpoints are saved in the specified directory as checkpoint_{epoch}. Make sure to backup them
# END Control variables----------------------------------

train_dataset = GF5BP(train)
validation_dataset = GF5BP(validation)
test_dataset = GF5BP(test)


# NETWORK INITIALIZATION
assert torch.cuda.is_available(), "Notebook is not configured properly!"
device = 'cuda:0'
print("Training network on {}".format(torch.cuda.get_device_name(device=device)))
net = Urnet(25).to(device)
num_params = sum([np.prod(p.shape) for p in net.parameters()])
print(f"Number of parameters : {num_params}")
print("Training samples: {}".format(train_dataset.__len__()))
print("Validation samples: {}".format(validation_dataset.__len__()))
print("Test samples: {}".format(test_dataset.__len__()), flush=True)

#Dataset train/validation split according to validation split and seed.


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = batch_size)
#for validation loader batch size is default, so 1.
crit = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.90, weight_decay=0.00001)


training_loss_values = []
validation_loss_values = []

macro_precision = []
weighted_precision = []
loss = 0 # pre-initialize loss to have global scope

if load_checkpoint != "":
    # Load model checkpoint (to resume training)
    checkpoint = torch.load(load_checkpoint)
    net.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    training_loss_values = checkpoint['training_loss_values']
    validation_loss_values = checkpoint['validation_loss_values']
    batch_size = checkpoint['batch_size']
    macro_precision = checkpoint['macro_precision']
    weighted_precision = checkpoint['weighted_precision']
else:
    last_epoch = 0

if not Path(checkpoint_directory).is_dir():
    print("Please provide a valid directory to save checkpoints in.")
else:
    for epoch in range(last_epoch, epochs):
        if verbose:
            pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}')
        net.train()
        print("Started epoch {}".format(epoch+1), flush=True)
        for batch_index, (image, mask) in enumerate(train_loader):
            image, mask = image.to(device), mask.to(device)
            mask_pred = net(image.type(torch.float32)).to(device)
            loss = crit(mask_pred, mask.squeeze().type(torch.long))
            training_loss_values.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            if verbose:
                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})
        if verbose:
            pbar.close()
        # run evaluation!
        # 1) Re-initialize data loaders
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)
        # 2) Call evaluation Loop (run model for 1 epoch on validation set)
        print("Running validation...", flush=True)
        validation_loss_values += validation_loss(net, validation_loader, crit, device, show_progress=False)
        # 3) Append results to list
        if (epoch+1) % freq == 0: # save checkpoint every freq epochs            
            save_model(epoch, net, opt, loss, training_loss_values, validation_loss_values, macro_precision, weighted_precision, batch_size, checkpoint_directory)
            print("Saved checkpoint {}".format(epoch+1))
        if (epoch+1) % precision_evaluation_freq == 0:
            print("Evaluating precision after epoch {}".format(epoch+1))
            precision_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = 1)
            macro, weighted = eval_model(net, precision_loader, device, show_progress=False)
            macro_precision.append(macro)
            weighted_precision.append(weighted)

    print("Training Done!")
    print(f"Reached training loss: {training_loss_values[-1]}")
    print(f"Reached validation loss: {validation_loss_values[-1]}")
