# This script is made to evaluate the performance of a model
import sys
import yaml
import utils
import dataset
import matplotlib.pyplot as plt

inFile = sys.argv[1]
with open(inFile,"r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
utils.display_configs(config)

device = utils.load_device(config)
net = utils.load_network(config, device).to(device)
if 'load_checkpoint' in config.keys(): utils.load_checkpoint(config, net)

# chunk size of 1 for validation and no random shift
loader = dataset.Loader(config['dataset'], 1, 
                        patch_size=config['patch_size'],
                        load_context=config['load_context'],
                        load_color_mask = config['load_color_mask'])
if 'images' in config.keys() : loader.specify(config['images'])
flat, normalized = utils.eval_model(net, 
                                    loader, 
                                    device, 
                                    1, 
                                    show_progress=config['verbose'],
                                    ignore_background=config['ignore_background'])
confusion = flat.compute()
utils.print_metrics(confusion)
# plot normalized confusion matrix
utils.plot_confusion(normalized, path=config.get('confusion_matrix'))
utils.plot_priors(confusion, path=config.get('priors'))