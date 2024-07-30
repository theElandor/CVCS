import torch
from pathlib import Path
from tqdm import tqdm
import utils
import yaml
import sys
inFile = sys.argv[1]

with open(inFile,"r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print("LOADED CONFIGURATIONS:")
print(config)
train_dataset, validation_dataset, test_dataset = utils.load_dataset(config)
device = utils.load_device(config)

try:
    net = utils.load_network(config, device)
except:
    print("Error in loading network.")
    exit(0)

utils.print_sizes(net, train_dataset, validation_dataset, test_dataset)

try:
    crit = utils.load_loss(config, device, train_dataset)
except:
    print("Error in loading loss module.")
    exit(0)
try:
    opt = utils.load_optimizer(config, net)
except:
    print("Error in loading optimizer")
    exit(0)


training_loss_values = []
validation_loss_values = []
macro_precision = []
weighted_precision = []

if  'load_checkpoint' in config.keys():
    # Load model checkpoint (to resume training)    
    checkpoint = torch.load(config['load_checkpoint'])
    net.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']+1
    training_loss_values = checkpoint['training_loss_values']
    validation_loss_values = checkpoint['validation_loss_values']
    config['batch_size'] = checkpoint['batch_size']
    macro_precision = checkpoint['macro_precision']
    weighted_precision = checkpoint['weighted_precision']
    print("Loaded checkpoint {}".format(config['load_checkpoint']), flush=True)
else:
    last_epoch = 0

if not Path(config['checkpoint_directory']).is_dir():
    print("Please provide a valid directory to save checkpoints in.")
else:    
    for epoch in range(last_epoch, config['epochs']):        
        print("Started epoch {}".format(epoch+1), flush=True)
        #initialize train_loader at each epoch to have a different shuffle every time
        train_loader = utils.load_loader(train_dataset, config, True)
        if config['verbose']:
            pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}')
        net.train()
        for batch_index, (image, mask, context) in enumerate(train_loader):
            image, mask = image.to(device), mask.to(device)
            # avoid loading context to GPU if not needed
            if net.requires_context:
                context = context.to(device)            
            mask_pred = net(image.type(torch.float32), context.type(torch.float32)).to(device)
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
        validation_loader = utils.load_loader(validation_dataset, config, False)
        # 2) Call evaluation Loop (run model for 1 epoch on validation set)
        print("Running validation...", flush=True)
        validation_loss_values += utils.validation_loss(net, validation_loader, crit, device, show_progress=config['verbose'])
        # 3) Append results to list

        if (epoch+1) % config['precision_evaluation_freq'] == 0:
            print("Evaluating precision after epoch {}".format(epoch+1), flush=True)
            precision_loader = utils.load_loader(validation_dataset, config, False, batch_size=1)
            macro, weighted = utils.eval_model(net, precision_loader, device, show_progress=config['verbose'])
            print(f"mIou: {macro}")
            print(f"weighted mIoU: {weighted}", flush=True)
            macro_precision.append(macro)
            weighted_precision.append(weighted)

        if (epoch+1) % config['freq'] == 0: # save checkpoint every freq epochs            
            utils.save_model(epoch, net, opt, training_loss_values, validation_loss_values, macro_precision, weighted_precision, 
                       config['batch_size'], 
                       config['checkpoint_directory'], 
                       config['opt']
                    )
            print("Saved checkpoint {}".format(epoch+1), flush=True)

    print("Training Done!")
    print(f"Reached training loss: {training_loss_values[-1]}")
    print(f"Reached validation loss: {validation_loss_values[-1]}")
