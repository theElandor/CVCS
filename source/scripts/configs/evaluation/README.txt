Most of the configuration file parameters are alredy
explained in source/scripts/configs/train.
Here I report only the ones specific for this script:

+ confusion_matrix: <path>
    path of the output file where to save
    the normalized confusion matrix;
+ priors: <path>
    path where to save the class priors plot;
+ images: <list of int>
    specify indexes of images that you want to use for evaluation.
    It is usefull when debugging to shrink the evaluation set to 
    just a couple of images.
    For example,
    images: [0,1,2] will evaluate the model on just the first 3 
    full-sized images of the specified dataset.
-----------------------------------------------
Using the Ensemble:
You can test an ensemble of networks by just specifying the desired
networks and relative checkpoints in a yaml file in the configs\ensemble\
directory. For example, this is an ensemble of 3 networks:

Unetv2: "D:\\weights\\checkpoint8"
Resnet101: "D:\\weights\\checkpoint12"
MobileNet: "D:\\weights\\checkpoint20"

In the evaluation configuration file, just specify

net: Ensemble
ensemble_config: 'test.yaml' # name of your config file

The ensemble class will automatically look for your ensemble configuration
file in the config\ensemble directory, and warn you if it does not find it.
The ensemble uses majority voting on the predicted class indexes (not on the logits obv.)