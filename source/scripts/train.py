import torch
import torch.nn as nn
import torchvision.transforms as v2
import numpy as np
from torch.utils.data import ConcatDataset
from dataset import PostDamDataset
from nets import Tunet
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
from tqdm import tqdm
from utils import validation_loss
import matplotlib.pyplot as plt
import os

# START Control variables---------------------------------
# This section contains some variables that need to be set before running the script.
images_path = "/work/cvcs2024/MSseg/Postdam_300x300_full/Images/"
labels_path = "/work/cvcs2024/MSseg/Postdam_300x300_full/Labels"
checkpoint_directory = "/homes/mlugli/checkpoints/tunet1/" # directory to save checkpoints in
extension = ".png" # extension of output files if produced.
epochs = 40
load_checkpoint = "/homes/mlugli/checkpoints/tunet1/checkpoint30"
freq = 5 # checkpoint saving frequency. If set to 2, it will save a checkpoint every 2 epochs.
# checkpoints are saved in the specified directory as checkpoint_{epoch}. Make sure to backup them
# END Control variables----------------------------------



transforms = v2.Compose([
    v2.GaussianBlur(kernel_size=(15), sigma=5),
    v2.ElasticTransform(alpha=200.0)
])

base_dataset = PostDamDataset(images_path, labels_path, extension)
augmented_dataset = PostDamDataset(images_path, labels_path,extension, transforms=transforms)
dataset = ConcatDataset([base_dataset, augmented_dataset])

# NETWORK INITIALIZATION
assert torch.cuda.is_available(), "Notebook is not configured properly!"
device = 'cuda:0'
print("Training network on {}".format(torch.cuda.get_device_name(device=device)))
net = Tunet(768, 12, 12).to(device)
num_params = sum([np.prod(p.shape) for p in net.parameters()])
print(f"Number of parameters : {num_params}")
print("Dataset length: {}".format(dataset.__len__()))

#Dataset train/validation split according to validation split and seed.
batch_size = 4
validation_split = .2
random_seed= 42

dataset_size = len(dataset)
base_indices = list(range(dataset_size//2))
np.random.seed(random_seed)
np.random.shuffle(base_indices)
augmented_indices = [i+(len(dataset)//2) for i in base_indices] # take coresponding augmented images
split = int(np.floor((1-validation_split) * (dataset_size//2)))

train_indices = base_indices[:split]+augmented_indices[:split]
val_base_indices = base_indices[split:]
val_noisy_indices = augmented_indices[split:]

train_sampler = SubsetRandomSampler(train_indices)
valid_base_sampler = SubsetRandomSampler(val_base_indices)
valid_noisy_sampler = SubsetRandomSampler(val_noisy_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
#for validation loader batch size is default, so 1.
validation_base_loader = torch.utils.data.DataLoader(dataset ,sampler=valid_base_sampler)
validation_noisy_loader = torch.utils.data.DataLoader(dataset ,sampler=valid_noisy_sampler)

print(f"Train dataset split(augmented): {len(train_indices)}")
print(f"Validation dataset split: {len(val_base_indices)}")
print(f"Validation dataset split(only noise): {len(val_noisy_indices)}")
crit = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.999)


training_loss_values = []
validation_loss_values = []
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
else:
    last_epoch = 0

if not Path(checkpoint_directory).is_dir():
    print("Please provide a valid directory to save checkpoints in.")
else:
    for epoch in range(last_epoch, epochs):
        cumulative_loss = 0
        tot = 0
        net.train()
        print("Started epoch {}".format(epoch+1), flush=True)
        for batch_index, (image, mask, _) in enumerate(train_loader):
            tot+=1
            image, mask = image.to(device), mask.to(device)        
            mask_pred = net(image).to(device)
            loss = crit(mask_pred, mask)
            cumulative_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        training_loss_values.append(cumulative_loss/tot)        
        # run evaluation!
        # 1) Re-initialize data loaders
        valid_base_sampler = SubsetRandomSampler(val_base_indices)
        validation_base_loader = torch.utils.data.DataLoader(dataset ,sampler=valid_base_sampler, batch_size=batch_size)
        # 2) Call evaluation Loop (run model for 1 epoch on validation set)
        print("Running validation...", flush=True)
        validation_loss_values.append(validation_loss(net, validation_base_loader, crit, device))
        # 3) Append results to list
        if (epoch+1) % freq == 0: # save checkpoint every 2 epochs
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss.item(),
                'training_loss_values': training_loss_values,
                'validation_loss_values': validation_loss_values
                }, os.path.join(checkpoint_directory, "checkpoint{}".format(epoch+1)))
    print("Training Done!")
    # make sure to save at the end of the training
    torch.save({
                'epoch': epochs,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss.item(),
                'training_loss_values': training_loss_values,
                'validation_loss_values': validation_loss_values
                }, os.path.join(checkpoint_directory, "checkpoint{}".format(epochs)))
    plt.plot(training_loss_values, label="Mean training loss per epoch")
    plt.plot(validation_loss_values, label = "Mean validation loss per epoch")
    plt.legend(loc="upper right")
    plt.savefig("train_validation_loss.png")
