import torch
from pathlib import Path
from tqdm import tqdm
import utils
import yaml
import sys
import random
import dataset, dataset2
from datetime import datetime
from prettytable import PrettyTable
import traceback
from torch.nn import DataParallel

inFile = sys.argv[1]

with open(inFile, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
utils.display_configs(config)

# image_transforms, mask_transforms = utils.load_basic_transforms(config)

# Loader_train = dataset.Loader(config['train'],
#                               config['chunk_size'],
#                               random_shift=True,
#                               patch_size=config['patch_size'],
#                               image_transforms=image_transforms,
#                               mask_transforms=mask_transforms)
# Loader_validation = dataset.Loader(config['validation'],
#                                    1,
#                                    patch_size=config['patch_size'])

image_transforms, geometric_transform = utils.load_basic_transforms(config)
Loader_train = dataset2.GID15(config['train'], 224, chunk_size=config['chunk_size'], dict_layout=False,
                              image_transforms=image_transforms, geometric_transforms=geometric_transform,
                              scale_down=True)
Loader_validation = dataset2.GID15(config['validation'],
                                   config['patch_size'],
                                   1)

# if config.get('debug'):
#     Loader_train.specify([0, 1])  # debug, train on 2 images only
#     Loader_validation.specify([0])  # debug, validate on 1 image only

device = utils.load_device(config)

t = PrettyTable(['Name', 'Value'])
try:
    net = utils.load_network(config, device)
    if config['parallel']:
        net = DataParallel(net)
        for i in range(torch.cuda.device_count()):
            t.add_row([f"GPU{i}", torch.cuda.get_device_name(i)])
        device = "cuda" if device.type.__contains__("cuda") else device
    t.add_row(['parameters', utils.count_params(net)])
except:
    traceback.print_exc()
    print("Error in loading network.")
    exit(0)

# t.add_row(['Patch size', Loader_train.patch_size])
# t.add_row(['Tpe', Loader_train.tpi])
# t.add_row(['Training patches', len(Loader_train.idxs) * Loader_train.tpi])
# t.add_row(['Validation patches', len(Loader_validation.idxs) * Loader_validation.tpi])
t.add_row(['Patch size', Loader_train._patch_shape])
t.add_row(['Tpe', Loader_train._tpi])
t.add_row(['Training patches', len(Loader_train._files_idxs) * Loader_train._tpi])
t.add_row(['Validation patches', len(Loader_validation._files_idxs) * Loader_validation._tpi])
print(t, flush=True)

try:
    crit = utils.load_loss(config, device, Loader_train)
except:
    traceback.print_exc()
    print("Error in loading loss module.")
    exit(0)
try:
    opt = utils.load_optimizer(config, net)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(opt)
except:
    print("Error in loading optimizer")
    exit(0)

training_loss_values = []  # store every training loss value
validation_loss_values = []  # store every validation loss value
macro_precision = []  # store AmIoU after each epoch (DEPRECATED)
weighted_precision = []  # store AwIoU after each epoch (DEPRECATED)
conf_flat = []  # store unnormalized confusion matrix after each epoch
conf_normalized = []  # store normalized confusion matrix after each epoch

if 'load_checkpoint' in config.keys():
    # Load model checkpoint (to resume training)    
    checkpoint = torch.load(config['load_checkpoint'])
    if net.wrapper:
        net.custom_load(checkpoint)
    else:
        net.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    last_epoch = checkpoint['epoch'] + 1
    training_loss_values = checkpoint['training_loss_values']
    validation_loss_values = checkpoint['validation_loss_values']
    config['batch_size'] = checkpoint['batch_size']
    macro_precision = checkpoint['macro_precision']  # (DEPRECATED)
    weighted_precision = checkpoint['weighted_precision']  # (DEPRECATED)
    # try to load confusion matrix, usefull for retrocompatibility for old models.
    try:
        conf_flat = checkpoint['conf_flat']
        conf_normalized = checkpoint['conf_normalized']
    except:
        print("Cannot find confusion matrix in specified checkpoint.")
    print("Loaded checkpoint {}".format(config['load_checkpoint']), flush=True)
else:
    last_epoch = 0

assert Path(config['checkpoint_directory']).is_dir(), "Please provide a valid directory to save checkpoints in."

for epoch in range(last_epoch, config['epochs']):
    print("[{}]Started epoch {}".format(str(datetime.now().time())[:8], epoch + 1), flush=True)

    patch_size = random.choice([56, 112, 224])

    Loader_train.set_patch_size(patch_size)
    Loader_train.shuffle()
    dl = torch.utils.data.DataLoader(Loader_train, batch_size=config['batch_size'] * ((224 // patch_size) ** 2))
    loss = 0
    if config['verbose']:
        pbar = tqdm(total=len(dl), desc=f'Epoch {epoch + 1}')
    net.train()
    for batch_index, (image, index_mask, color_mask, context) in enumerate(dl):
        image, mask = image.to(device), index_mask.to(device)
        # avoid loading context to GPU if not needed
        if net.requires_context:
            context = context.to(device)

        mask_pred = net(image.type(torch.float32), context.type(torch.float32)).to(device)
        loss = crit(mask_pred, mask.squeeze(1).type(torch.long))

        training_loss_values.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if config['verbose']:
            pbar.update(1)
            pbar.set_postfix({'Loss': loss.item()})
    print("[{}]Last value of training loss at epoch {} was {}".format(str(datetime.now().time())[:8], epoch, loss))
    if config['verbose']:
        pbar.close()

    scheduler.step()
    print("Running validation...", flush=True)
    validation_loss_values += utils.validation_loss(net, Loader_validation, crit, device, config['batch_size'],
                                                    show_progress=config['verbose'])
    if (epoch + 1) % config['precision_evaluation_freq'] == 0:
        print("Evaluating precision after epoch {}".format(epoch + 1), flush=True)

        flat, normalized = utils.eval_model(net,
                                            Loader_validation,
                                            device,
                                            batch_size=1,
                                            show_progress=config['verbose'],
                                            ignore_background=config['ignore_background'])
        confusion = flat.compute()  # get confusion matrix as tensor
        utils.print_metrics(confusion)
        # keep track of precision and confusion matrix
        conf_flat.append(flat)
        conf_normalized.append(normalized)

    if (epoch + 1) % config['freq'] == 0:  # save checkpoint every freq epochs
        utils.save_model(epoch,
                         net, opt, scheduler,
                         training_loss_values, validation_loss_values,
                         macro_precision, weighted_precision,
                         conf_flat,
                         conf_normalized,
                         config['batch_size'],
                         config['checkpoint_directory'],
                         config['opt']
                         )
        print("Saved checkpoint {}".format(epoch + 1), flush=True)

print("Training Done!")
print(f"Reached training loss: {training_loss_values[-1]}")
print(f"Reached validation loss: {validation_loss_values[-1]}")
