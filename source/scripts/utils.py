import torch
from sklearn.metrics import jaccard_score as jsc
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from random import random
import torchvision.transforms as T

class Converter: # WARNING: POSTDAM CONVERTER
	def __init__(self):
		self.color_to_label = {
            (1, 1, 0): 0,  # Yellow (cars)
            (0, 1, 0): 1, # Green (trees)
            (0, 0, 1): 2, # Blue (buildings)
            (1, 0, 0): 3,  # Red (clutter)
            (1, 1, 1): 4, # White(impervious surface),
            (0, 1, 1): 5 # Aqua (low vegetation)
        }
	def iconvert(self, mask):
		"""
		Function needed to convert the class label mask needed by CrossEntropy Function
		to the original mask.
		input: class label mask, HxW
		output: original mask, HxWx3
		"""
		H,W = mask.shape
		colors = torch.tensor(list(self.color_to_label.keys())).type(torch.float64)
		labels = torch.tensor(list(self.color_to_label.values())).type(torch.float64)
		output = torch.ones(H,W,3).type(torch.float64)
		for color, label in zip(colors, labels):
			match = (mask == label)
			output[match] = color
		return output
	def convert(self,mask):
		"""
		Function needed to convert the RGB (Nx3x300x300) mask into a 
		'class label mask' needed when computing the loss function.
		In this new representation for each pixel we have a value
		between [0,C) where C is the number of classes, so 6 in this case.
		This new tensor will have shape Nx300x300.
		"""			
		C,H,W = mask.shape
		colors = torch.tensor(list(self.color_to_label.keys()))
		labels = torch.tensor(list(self.color_to_label.values()))
		reshaped_mask = mask.permute(1, 2, 0).reshape(-1, 3)
		class_label_mask = torch.zeros(reshaped_mask.shape[0], dtype=torch.long)
		for color, label in zip(colors, labels):
			match = (reshaped_mask == color.type(torch.float64)).all(dim=1)
			class_label_mask[match] = label
		class_label_mask = class_label_mask.reshape(H,W)		
		return class_label_mask
      
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

def save_model(epoch, net, opt, loss, train_loss, val_loss, macro_precision, weighted_precision, batch_size, checkpoint_dir):
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