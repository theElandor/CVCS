import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import validation_loss, save_model, eval_model, load_network, load_gaofen, print_sizes, load_optimizer
import yaml
import sys
inFile = sys.argv[1]

with open(inFile,"r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print("LOADED CONFIGURATIONS:")
print(config)

train_dataset, validation_dataset, test_dataset = load_gaofen(config['train'], config['validation'], config['test'])

# NETWORK INITIALIZATION
assert torch.cuda.is_available(), "Notebook is not configured properly!"
device = 'cuda:0'
print("Training network on {}".format(torch.cuda.get_device_name(device=device)))

try:
    net = load_network(config['net']).to(device)
except:
    exit(0)

print_sizes(net, train_dataset, validation_dataset, test_dataset)

#Dataset train/validation split according to validation split and seed.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'])
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = config['batch_size'])
#for validation loader batch size is default, so 1.
crit = nn.CrossEntropyLoss()

try:
    opt = load_optimizer(config['opt'], net)
except:
    exit(0)


training_loss_values = []
validation_loss_values = []
macro_precision = []
weighted_precision = []

loss = 0 # pre-initialize loss to have global scope

if  'load_checkpoint' in config.keys():
    # Load model checkpoint (to resume training)    
    checkpoint = torch.load(config['load_checkpoint'])
    net.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    training_loss_values = checkpoint['training_loss_values']
    validation_loss_values = checkpoint['validation_loss_values']
    config['batch_size'] = checkpoint['batch_size']
    macro_precision = checkpoint['macro_precision']
    weighted_precision = checkpoint['weighted_precision']
else:
    last_epoch = 0

if not Path(config['checkpoint_directory']).is_dir():
    print("Please provide a valid directory to save checkpoints in.")
else:    
    for epoch in range(last_epoch, config['epochs']):        
        print("Started epoch {}".format(epoch+1))            
        if config['verbose']:            
            pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}')
        net.train()    
        for batch_index, (image, mask) in enumerate(train_loader):
            if batch_index == 3:
                break
            image, mask = image.to(device), mask.to(device)
            mask_pred = net(image.type(torch.float32)).to(device)
            loss = crit(mask_pred, mask.squeeze().type(torch.long))
            training_loss_values.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            if config['verbose']:
                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})
        if config['verbose']:
            pbar.close()
        # run evaluation!
        # 1) Re-initialize data loaders
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config['batch_size'])
        # 2) Call evaluation Loop (run model for 1 epoch on validation set)
        print("Running validation...", flush=True)
        #validation_loss_values += validation_loss(net, validation_loader, crit, device, show_progress=config['batch_size'])
        # 3) Append results to list

        if (epoch+1) % config['precision_evaluation_freq'] == 0:
            print("Evaluating precision after epoch {}".format(epoch+1))
            precision_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = 1)
            macro, weighted = eval_model(net, precision_loader, device, show_progress=config['val_progress'])
            macro_precision.append(macro)
            weighted_precision.append(weighted)        

        if (epoch+1) % config['freq'] == 0: # save checkpoint every freq epochs            
            save_model(epoch, net, opt, loss, training_loss_values, validation_loss_values, macro_precision, weighted_precision, 
                       config['batch_size'], 
                       config['checkpoint_directory'], 
                       config['opt']
                    )
            print("Saved checkpoint {}".format(epoch+1))

    print("Training Done!")
    print(f"Reached training loss: {training_loss_values[-1]}")
    print(f"Reached validation loss: {validation_loss_values[-1]}")
