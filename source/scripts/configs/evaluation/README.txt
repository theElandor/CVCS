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