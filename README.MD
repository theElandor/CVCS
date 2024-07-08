
# Table of Contents

1.  [Important Papers](#orgfb6d168)
    1.  [Datasets](#org7a0c276)



<a id="orgfb6d168"></a>

# Important Papers

-   Mohammad Barr, Enhancing the ability of convolutional neural networks for remote
    sensing image segmentation using transformers <https://doi.org/10.1007/s00521-024-09743-6>
-   Original U-net paper (conv only): <https://doi.org/10.1007/978-3-319-24574-4_28>
-   Helber, P.; Bischke, B.; Dengel, A.; Borth, D. EuroSAT: a novel dataset and deep learning benchmark for
    land use and land cover classification. (Big dataset for land cover classification);
-   Transformers for Image Recognition at Scale: <https://doi.org/10.48550/arXiv.2010.11929>
-   Fatema-E- Jannat, Andrew R. Willis;Improving Classification of Remotely Sensed
    Images with the Swin Transformer
-   Bazi, Y.; Bashmal, L.; Rahhal, M.M.A.; Dayil, R.A.; Ajlan, N.A. Vision Transformers for Remote Sensing Image Classification. Remote Sens. 2021, 13, 516. <https://doi.org/10.3390/rs13030516>
-   MCAT: usefull as a survey and for comparison: <https://doi.org/10.1109/JSTARS.2024.3397488>
-   "Cross shaped windows": CSWin Transformer: A General Vision Transformer Backbone With Cross-Shaped Windows


<a id="org7a0c276"></a>

# Datasets

-   **EuroSat**: <https://github.com/phelber/eurosat>
-   **Postdam**, 6000x6000 Images:
    Classes:
    
    1.  Yellow: cars;
    2.  Green: trees;
    3.  Blue: buildings;
    4.  Red : clutter;
    5.  White : impervious surface;
    6.  Aqua : low vegetaion;
    
    <https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx#Potsdam>
- PostDam, 300x300 cropped data, RGB masks --> https://drive.google.com/drive/folders/1C0s_fMhjgz0aXucyYv8BZqJnPzBIaHSA?usp=drive_link
-   **DeepGlobe** (road extraction): 1024x1024 images with segmentation masks.
    <https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset>
# Research Notes
- https://docs.google.com/spreadsheets/d/14Pbsek03Q953QFlza4xGU4S9osLEyEKtXywEvMGKYsU/edit?usp=sharing
# Things to do
+ Make a 256x256 variant of Postdam (currently using 300x300 images);
+ Access Unimore servers and launch a sample training;
+ Add transformer to Unet architecture and train on postdam;
+ ~~Write evaluation loop with different metrics~~;
+ Write reliable code to make checkpoints of model and optimizer every 50(?) epochs, so that we can plot
  performance over epochs of training;
+ Improve GPU usage (bottleneck somewhere);
