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
try:
    net = utils.load_network(config, device).to(device)
except:
    print("Error in loading network.")
    exit(0)
if 'load_checkpoint' in config:
    utils.load_checkpoint(config, net)
# chunk size of 1 for validation and no random shift
loader = dataset.Loader(config['dataset'], 1, patch_size=config['patch_size'])
if 'images' in config.keys():
    loader.specify(config['images'])
AmIoU, AwIoU, flat, normalized = utils.eval_model(net, loader, device, 1, show_progress=config['verbose'])
confusion = flat.compute()
utils.print_metrics(AmIoU, AwIoU, confusion)
# plot normalized confusion matrix
utils.plot_confusion(normalized, path=config.get('confusion_matrix'))
utils.plot_priors(confusion, path=config.get('priors'))