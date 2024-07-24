import torch
from sklearn.metrics import jaccard_score as jsc
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from random import random
import torchvision.transforms as T
import nets
from dataset import GF5BP
import numpy as np

def eval_model(net, validation_loader, device, show_progress = False):
    # returns (macro, weighted) IoU
    macro = 0
    weighted = 0
    if show_progress:
        pbar = tqdm(total=len(validation_loader.dataset))
    with torch.no_grad():
        net.eval()
        for x,y in validation_loader:
            x, y = x.to(device), y.to(device)
            y_pred = net(x.type(torch.float32))            
            y_pred = y_pred.squeeze().cpu()
            _,pred_mask = torch.max(y_pred, dim=0)

            prediction = pred_mask.cpu().numpy().reshape(-1)
            target = y.cpu().numpy().reshape(-1)        
            weighted += jsc(target,prediction, average='weighted') # takes into account label imbalance
            macro += jsc(target,prediction, average='macro') # simple mean over each class.                        
            if show_progress:
                pbar.update(1)
    macro_score = macro / len(validation_loader.dataset)
    weighted_score = weighted / len(validation_loader.dataset)
    if show_progress:
        pbar.close()
    print("Macro IoU score: {}".format(macro_score))        
    print("Weigthed IoU score: {}".format(weighted_score))
    return macro_score, weighted_score
    
    # validation_loss(net, validation_base_loader, len(val_base_indices))
def validation_loss(net, loader, crit, device, show_progress = False):
    loss_values = []
    net.eval()
    if show_progress:
        pbar = tqdm(total=len(loader))
    with torch.no_grad():
        for image, mask in loader:
            image, mask = image.to(device), mask.to(device)        
            mask_pred = net(image.type(torch.float32)).to(device)
            loss = crit(mask_pred, mask.squeeze().type(torch.long))
            loss_values.append(loss.item())
            if show_progress:
                pbar.update(1)
    if show_progress:
        pbar.close()
    return loss_values

def save_loss(filename, values):
    with open(filename, "w") as f:
        for v in values:
            f.write(str(v)+"\n")

def save_model(epoch, net, opt, loss, train_loss, val_loss, macro_precision, weighted_precision, batch_size, checkpoint_dir, optimizer):
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss.item(),
        'training_loss_values': train_loss,
        'validation_loss_values': val_loss,
        'batch_size': batch_size,
        'macro_precision': macro_precision, 
        'weighted_precision': weighted_precision,
        'optimizer': optimizer,
        }, os.path.join(checkpoint_dir, "checkpoint{}".format(epoch+1)))
     

class RandomFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask):
        if random() < self.prob:
            # Apply horizontal flip
            img = T.functional.hflip(img)
            mask = T.functional.hflip(mask)
        if random() < self.prob:
            # Apply vertical flip
            img = T.functional.vflip(img)
            mask = T.functional.vflip(mask)
        return img, mask
    
def inference(net, dataset, indexes, device):
    net.eval()
    with torch.no_grad():
        for index in indexes:
            image, mask = dataset[index]
            image, mask = image.to(device), mask.to(device)
            output = net(image.unsqueeze(0).type(torch.float32))
            f, axarr = plt.subplots(1,3)
            axarr[0].imshow(image.permute(1,2,0).cpu())
            axarr[1].imshow(mask.permute(1,2,0).cpu())
            axarr[2].imshow(torch.argmax(output.squeeze().permute(1,2,0).cpu(), dim=2))
            plt.savefig(os.path.join("output", f"{index}.png"))

def load_network(netname):
    if netname == 'TSwin':
        return nets.Swin(96,224,25)
    elif netname == 'BSwin':
        return nets.Swin(128,224,25)
    elif netname == 'Unet':
        return nets.Urnet(25)
    else:
        raise ValueError("Invalid network name.")
    

def load_gaofen(train, validation, test):    
    train_dataset = GF5BP(train)
    validation_dataset = GF5BP(validation)
    test_dataset = GF5BP(test)
    return train_dataset, validation_dataset, test_dataset
     
def print_sizes(net, train_dataset, validation_dataset, test_dataset):
    num_params = sum([np.prod(p.shape) for p in net.parameters()])
    print(f"Number of parameters : {num_params}")
    print("Training samples: {}".format(train_dataset.__len__()))
    print("Validation samples: {}".format(validation_dataset.__len__()))
    print("Test samples: {}".format(test_dataset.__len__()), flush=True)

def load_optimizer(optimizer, net):
    if optimizer == 'SGD1':
        return torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.90, weight_decay=0.00001)
    elif optimizer == 'ADAM1':
        return torch.optim.Adam(net.parameters(), lr=5e-4)
    else:
        raise ValueError("Optimizer name not valid.")