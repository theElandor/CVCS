+ train: <path>
  Training samples path

+ validation: <path>
  Validation samples path

+ test: <path>
  Test samples path

+ checkpoint_directory: <path>
  directory to save checkpoints in

+ load_checkpoint: <path>
  if specified, the training script wil try to load a checkpoint

+ device: <str>
    1) "gpu" if it is available
    2) "cpu" to debug on cpu

+ epochs: <int>
  number of epochs (usually 10 to 20)
  warning: if you load a checkpoint, make sure to specify the total number of 
  epochs you want. For example, if the model alredy trained for 10 epochs and 
  you want to train for 10 more, then specify epochs: 20

+ chunk size: <int>
  number of full-sized images that are loaded in RAM at once and shuffled during
  training. On remote server you can set this from 10 to 20.

+ freq: <int>
  checkpoint saving frequency. If set to 2, a checkpoint will be saved every 2 epochs.  

+ verbose: <bool>
  show progress bar during epochs. Keep false if you train on remote server

+ batch_size: <int>
  this parameter will be overwritten if a checkpoint is loaded

+ precision_evaluation_freq: <int>
  if set to 2, it will evaluate model precision once every 2 epochs

+ net: <str>
  Avaliable models:
  TSwin: Tiny swin transformer backbone (hugging face) + unet decoder with residuals
  BSwin: Base swin transformer backbone (hugging face) + unet decoder with residuals
  Unet: Classic unet network (2015 medical segmentation paper)
  Unetv2: Unet with Transpose Convolution in decoding path
  Unet_torch: unet from torch repository (heavy, 300M parameters(?))
  FUnet: Fusion network, Unet with double path (testing)
  Resnet101: Resnet101 trained with DeepLabv3

+ opt: <str>
  Available setups:
  1) SGD1
  2) ADAM1

+ loss: <str>
  Available losses:
  1) DEL: DiceEntropyLoss
  2) CEL: CrossEntropyLoss
  3) DL: DiceLoss
  
+ num_classes: <int>
  For GID15 dataset, keep this set to 15

+ patch_size: <int>
  Size of patch cropped from original sized image.
  This patch size will be used in training, validation and evaluation loops

+ random_tps: <list of int>
  example: [[512,0.5], [1024, 1]]
  In this example we "augment" each chunk with random cropped 512x512 and 1024x1024 patches
  that are then rescaled to default patch size. We call these patches "augmented patches".
  You can specify for each size how many augmented patches you want to add: in this case,
  we add 50% of the original dataset size of 512x512 augmented patches and 100% of the
  original dataset size of 1024x1024 augmented patches.
  Context is cropped and returned accordingly, following the same rules of standard patches.

+ augmentation: <bool>
  If set to True, some basic data augmentation random
  transforms will be passed to the training data loader.

+ debug_plot: <bool>
  If set to True, the training procedure will call the 
  debug_plot function at each iteration.
  The function plots the path, color mask and context
  of the first sample found in the batch.
  Made for debug purposes.


+ ignore_background <bool>
  Wether or not to ignore the background class (index = 0)
  Background will not be considered even when evaluating the model
  during training.

+ load_context <bool>
  Set this to True if your network requires context around the patch.
  If set to False, the data loader will not crop and save Context
  to optimize performance and memory uage.

+ load_color_mask <bool>
  Set this to True if you need the color_mask for plotting purposes.
  If set to False, the data loader will not crop and save the
  color mask to optimize performance and memory usage.