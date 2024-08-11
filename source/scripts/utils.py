import torch
from sklearn.metrics import jaccard_score as jsc
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from random import random
import torchvision.transforms as T
import nets
from dataset import GID15
import numpy as np
import torch.nn as nn
import loss
from PIL import Image

def eval_model(net, Loader_validation, device, batch_size=1, show_progress=False):
    # returns (macro, weighted) IoU
    macro = 0
    weighted = 0    
    net.eval()
    with torch.no_grad():
        for c in range(len(Loader_validation)): 
            dataset = Loader_validation.get_iterable_chunk(c)
            dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
            if show_progress:
                pbar = tqdm(total=len(dataset.chunk_crops)//batch_size, desc=f'Chunk {c+1}')
            for x,y,_,context in dl:
                x, y = x.to(device), y.to(device)
                if net.requires_context:
                    context = context.to(device)
                y_pred = net(x.type(torch.float32), context.type(torch.float32))
                y_pred = y_pred.squeeze().cpu()
                _,pred_mask = torch.max(y_pred, dim=0)

                prediction = pred_mask.cpu().numpy().reshape(-1)
                target = y.cpu().numpy().reshape(-1)        
                weighted += jsc(target,prediction, average='weighted') # takes into account label imbalance
                macro += jsc(target,prediction, average='macro') # simple mean over each class.                        
                if show_progress:
                    pbar.update(1)                
            if show_progress:
                pbar.close()
    macro_score = macro / ((len(dataset.chunk_crops)*len(Loader_validation))// batch_size)
    weighted_score = weighted / ((len(dataset.chunk_crops)*len(Loader_validation))// batch_size)
    if show_progress:
        pbar.close()
    return macro_score, weighted_score
    
    # validation_loss(net, validation_base_loader, len(val_base_indices))
def validation_loss(net, Loader_validation, crit, device, bs, show_progress=False):
    loss_values = []
    net.eval()    
    with torch.no_grad():
        for c in range(len(Loader_validation)):
            dataset = Loader_validation.get_iterable_chunk(c)
            dl = torch.utils.data.DataLoader(dataset, batch_size=bs)
            if show_progress:
                pbar = tqdm(total=len(dataset.chunk_crops)//bs, desc=f'Chunk {c+1}')
            for batch_index, (image, index_mask, _, context) in enumerate(dl):
                image, mask = image.to(device), index_mask.to(device)
                if net.requires_context:
                    context = context.to(device)
                mask_pred = net(image.type(torch.float32), context.type(torch.float32)).to(device)
                loss = crit(mask_pred, mask.squeeze().type(torch.long))
                loss_values.append(loss.item())
                if show_progress:
                    pbar.update(1)                
            if show_progress:
                pbar.close()
    return loss_values

def save_model(epoch, net, opt,scheduler, train_loss, val_loss, macro_precision, weighted_precision, batch_size, checkpoint_dir, optimizer):
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
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
    
def inference(net, dataset, indexes, device, converter, mask_only=False):
    net.eval()
    with torch.no_grad():
        for index in indexes:
            image,_, mask = dataset[index]
            image, mask = image.to(device), mask.to(device)
            output = net(image.unsqueeze(0).type(torch.float32))            
            pred_index = torch.argmax(output.squeeze().permute(1,2,0).cpu(), dim=2)         
            if not mask_only:
                f, axarr = plt.subplots(1,3)
                axarr[0].imshow(image.permute(1,2,0).cpu())
                axarr[1].imshow(mask.permute(1,2,0).cpu())
                axarr[2].imshow(converter.iconvert(pred_index))
                plt.savefig(os.path.join("output", f"{index}.png"))
            else:
                tensor = converter.iconvert(pred_index)
                arr = ((tensor*255).type(torch.uint8)).numpy()
                image = Image.fromarray(arr)
                image.save(os.path.join("output", f"{index}.png"))
                

def load_network(config, device):
    netname = config['net']
    classes = config['num_classes']
    if netname == 'TSwin':
        return nets.Swin(96,224,classes+1, device).to(device)
    elif netname == 'BSwin':
        return nets.Swin(128,224,classes+1, device).to(device)
    elif netname == 'Unet':
        return nets.Urnet(classes+1).to(device)
    elif netname == 'Fusion':
        return nets.Fusion(classes+1, device).to(device)
    elif netname == 'Unet_torch':
        return nets.UnetTorch(device)
    elif netname == 'Unetv2':
        return nets.Urnetv2(classes+1).to(device)
    elif netname == 'FUnet':
        return nets.FUnet(classes+1).to(device)
    else:
        print("Invalid network name.")
        raise Exception
    

def load_gaofen(train, validation, test):    
    train_dataset = GID15(train, random_shift=True)
    validation_dataset = GID15(validation)
    test_dataset = GID15(test)
    return train_dataset, validation_dataset, test_dataset

def count_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def load_optimizer(config, net):
    optimizer = config['opt']
    if optimizer == 'SGD1':
        return torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.90, weight_decay=0.0005)
    elif optimizer == 'ADAM1':
        return torch.optim.Adam(net.parameters(), lr=1e-4)
    else:
        raise ValueError("Optimizer name not valid.")

def load_loss(config, device, dataset=None):
    classes = config['num_classes']
    name = config['loss']
    if name == "CEL":
        return nn.CrossEntropyLoss()
    elif name == "DEL":        
        return loss.DiceEntropyLoss(device, classes)
    elif name == "wCEL":
        print("Computing class weights, it might take several minutes...")
        weights = dataset.get_class_weights().to(device)
        return nn.CrossEntropyLoss(weight=weights)
    else:
        raise Exception
    
def load_dataset(config):
    return load_gaofen(config['train'], config['validation'], config['test'])

def custom_shuffle(dataset):
    tpe = dataset.tiles_per_img
    images = [_ for _ in range(len(dataset)//tpe)]
    np.random.shuffle(images)
    ranges = []
    for image in images:
        r = [_ for _ in range(image*tpe, (image+1)*tpe)]
        np.random.shuffle(r)
        ranges.append(r)    
    final = sum(ranges, [])    
    return final

def load_loader(dataset, config, shuffle, batch_size=-1):
    if batch_size != -1:
        bs = batch_size
    else:        
        bs = config['batch_size']    
    assert dataset.tiles_per_img % bs == 0, "Tiles per image is not divisible by batch size, unexpected behaviour of DataLoader."    
    if shuffle:
        sampler=custom_shuffle(dataset)
    else:
        sampler=None
    loader = loader = torch.utils.data.DataLoader(dataset, batch_size=bs, sampler=sampler, shuffle=True)
    return loader


def load_device(config):
    if config['device'] == 'gpu':
        assert torch.cuda.is_available(), "Notebook is not configured properly!"
        device = 'cuda:0'
    else:
        device = torch.device('cpu')
    print("Training network on {}".format(torch.cuda.get_device_name(device=device)))
    return device

def load_checkpoint(config, net=None):
    if  'load_checkpoint' in config.keys():
    # Load model checkpoint (to resume training)    
        checkpoint = torch.load(config['load_checkpoint'])
        if net:
            net.load_state_dict(checkpoint['model_state_dict'])                
        TL = checkpoint['training_loss_values']
        VL = checkpoint['validation_loss_values']        
        mIoU = checkpoint['macro_precision']
        wIoU = checkpoint['weighted_precision']
        print("Loaded checkpoint {}".format(config['load_checkpoint']), flush=True)        
        print(f"mIoU: {mIoU}")
        print(f"wIoU: {wIoU}")
        return TL, VL, mIoU, wIoU