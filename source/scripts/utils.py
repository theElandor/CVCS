import torch
from sklearn.metrics import jaccard_score as jsc
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from random import random
import torchvision.transforms as T

class Converter:
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
      
def eval_model(net, validation_loader, validation_len, device, dataset, show_progress = False, write_output=False, prefix=""):
    # returns (macro, weighted) IoU
    c = Converter()
    macro = 0
    weighted = 0
    if show_progress:
        pbar = tqdm(total=validation_len)
    with torch.no_grad():
        net.eval()
        for i, (x,y, index) in enumerate(validation_loader):
            x, y = x.to(device), y.to(device)
            y_pred = net(x)
            x_ref = x.cpu()
            y_pred = y_pred.squeeze().cpu()
            _,pred_mask = torch.max(y_pred, dim=0)

            prediction = pred_mask.cpu().numpy().reshape(-1)
            target = y.cpu().numpy().reshape(-1)        
            weighted += jsc(target,prediction, average='weighted') # takes into account label imbalance
            macro += jsc(target,prediction, average='macro') # simple mean over each class.            
            if(write_output):
                fig ,axarr = plt.subplots(1,3)
                _,target_transformed_mask,_ = dataset.__getitem__(index.item())

                axarr[0].title.set_text('Original Image')
                axarr[0].imshow(x_ref.squeeze().swapaxes(0,2).swapaxes(0,1))

                axarr[1].title.set_text('Model Output')
                axarr[1].imshow(c.iconvert(pred_mask))

                axarr[2].title.set_text('Original Mask')
                axarr[2].imshow(c.iconvert(target_transformed_mask))
                plt.savefig(os.path.join("output", "{}_Image{}.png".format(prefix, i)))
                plt.close(fig)
            if show_progress:
                pbar.update(1)
    macro_score = macro / validation_len
    weighted_score = weighted / validation_len
    if show_progress:
        pbar.close()
    if(write_output):
        print("Macro IoU score: {}".format(macro_score))        
        print("Weigthed IoU score: {}".format(weighted_score))
    return macro_score, weighted_score
    
    # validation_loss(net, validation_base_loader, len(val_base_indices))
def validation_loss(net, loader, crit, device):
    cumulative_loss = 0
    iterations = 0
    net.eval()
    with torch.no_grad():
        for batch_index, (image, mask, _) in enumerate(loader):
            iterations+=1
            image, mask = image.to(device), mask.to(device)        
            mask_pred = net(image).to(device)
            loss = crit(mask_pred, mask)
            cumulative_loss += loss.item()
    return cumulative_loss/iterations

def save_loss(filename, values):
    with open(filename, "w") as f:
        for v in values:
            f.write(str(v)+"\n")

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