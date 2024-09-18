# GID15 Virtual Lab
This repo is an open source virtual laboratory to work on remote sensing
image segmentation. It features easy to edit code and scripts to
load state of the art neural networks and train them on datasets like GID15
https://captain-whu.github.io/GID15/.
This projects aims at keeping the code simple and easy to read, so that even unexpert
students can jump in and modify the code according to their needs.
This virtual laboratory revolves around 4 main files:
1) train.py: this files is the entry point if you want to customize your training procedure;
2) utils.py: just a file with some utility functions;
3) nets.py: file where you can declare and add new networks;
3) inference.py: script used to produce segmentation masks once you trained your model.
4) evaluation.py: script used to evaluate the performance of your model.

# Dependencies
```txt
torch==2.3.1
pillow
matplotlib
PyYAML
transformers
tqdm
prettytable
scikit-learn
numpy
torchmetrics
pandas
seaborn
```

# Quickstart
In this tutorial we show how to setup the training of a simple network.
You will also learn how to create a config file to customize hyperparameters and other settings.

## 1. Write the training config
We implemented several functions to cover most of the needs related to remote sensing image segmentation. The following is just an example, but you can find more information in the full documentation.
```yaml
#-------------DEBUGGING-----------
debug: False
debug_plot: False
verbose: True
#-------------DIRECTORIES------------
train: D:\Datasets\GID15\GID15\Train
validation: D:\Datasets\GID15\GID15\Validation
test: D:\Datasets\GID15\GID15\Test
checkpoint_directory: D:\weights\test\balanced
#--------DEVICE--------------
device: gpu
#----------NET---------------
net: Unetv2
#-------DATA LOADING----------
epochs: 50
chunk_size: 10
validation_chunk_size: 5
patch_size: 224
batch_size: 10
augmentation: True
# ------LOSS AND OPTIMIZATION----------
loss: CEL
opt: SGD2
ignore_background: True
#-------ALMOST NEVER TOUCH THESE--------
freq: 1
precision_evaluation_freq: 1
num_classes: 15
load_color_mask: False
load_context: False
```
Most of the parameters are self-explanatory, here are some more information on the most specific ones:
+ *checkpoint directory*: directory where the script will save a checkpoint of the model after each epoch.
+ *chunk size*: the default dataloader used for training does not require images to be divided in patches, as it will take care of cropping the original sized images at runtime. With this setting you can specify how many original sized images are loaded in memory, shuffled and cropped to obtain the training patches. Ideally you should set *chunk_size* as the number of full size images that you train on to maximize variability between patches.During training the dataloader will take care to iterate on the full dataset by creating *chunks* of the specified size.

+ *validation_chunk_size*: same concept as *chunk_size* but for the validation dataset.
+ *augmentation*: if set to True, it will apply some basic data augmentation tecniques to
the training patches, like random gaussian blur, random rotation and color jitter.
For more information on training parameters, refer to the full documentation.

## 2. Launch the training
To launch the training, use the following command
```bash
python3 train.py <config file>
```
## 3. Inference
Once your model finished the training procedure, you will immediately get a grasp on how it performed on the training dataset by looking at the standard output.
To perform inference and segment actual images, you can use the inference.py script paired with a configuration file. Here's an example:
```yaml
dataset: D:\Datasets\GID15\gid-15\GID\Validation
device: gpu
net: Unetv2
load_checkpoint: D:\weights\test\balanced\checkpoint50
patch_size: 224
border_correction: 256
range: [0,960]
mask_only: True
out_image: "map.png"
num_classes: 15
```
+ *load_checkpoint*: used to specify the checkpoint to load for your network. Make sure that *net* and the desired checkpoints are compatible, otherwise the script will throw an error.
+ *border_correction*: used to specify the actual size of patches that are going to be segmented. This tecnique is used to prevent 'tiling', a common problem in remote sensing image segmentation that we further explain in our paper. 
In practice, the dataloader will divide the image into non overlapping patches of size *patch_size*, but will perform inference on a 'augmented patch' of size *border_correction* then crop the central part to only keep *patch_size* x *patch_size* pixels.
+ *range*: with this list you can select the patches that you want to segment. In this example we selected the first 960 patches that corespond to the first full-sized image of the validation set. Every segmented patch will be saved in the ```output``` directory.
+ *mask_only*: if set to True, the output will be saved as just the segmentation mask in a .png file. If set to False, the output will be a plot that will show both the original image and coresponding mask.
+ *out_image*: if specified, the script will create a full-sized image with the generated masks.
+ **You can find the configuration files used to create full size images of the official paper with U-Net in source/scripts/configs/inference. To use them, make sure to download the checkpoint and the validation set.**
## 4. (extra): Evaluation
You can specify to run a evaluation loop (on the validation set) after each training epoch with the *eval* option in the training configuration file. Moreover, you can also run customized evaluation loops.
To do this,you will need a evaluation config file. Here's an example:
```yaml
net: Unetv2
load_checkpoint: D:\weights\test\balanced\UnetNoBackWeighted\checkpoint50
dataset: D:\Datasets\GID15\gid-15\GID\Validation
num_classes: 15
patch_size: 224
device: gpu
verbose: True
confusion_matrix: 'confusion.png'
priors: 'priors.png'
ignore_background: True
load_context: False
load_color_mask: False
``` 
with *confusion_matrix* and *priors* you can specify the path where the confusion matrix and the histogram of class priors of the specified dataset will be saved.