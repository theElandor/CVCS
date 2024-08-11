# create Train\ and Validation\
# Move all training images into Train\Image__8bit_NirRGB
# Move all validation images into Validation\Image__8bit_NirRGB

# Create Train\Annotation__index and Train\Annotation__color
# Create Validation\Annotation__index and Validation\Annotation__color

# Move all training images ending with .png to Train\Annotation
import os
import shutil
from pathlib import Path

def get_index(image):
    filename = Path(image).name
    return filename[:-4]+"_15label.png"
def get_mask(image):
    filename = Path(image).name    
    return filename[:-4]+"_15label.tif"
def copy(files, folder):
    for f in files:
        filename = Path(f).name
        shutil.move(f, os.path.join(folder, filename))
    


root = "D:\\Datasets\\GID15\\gid-15\\GID"
train_mask_path = os.path.join(root, "ann_dir", "train")
val_mask_path = os.path.join(root, "ann_dir", "val")

train_image_path = os.path.join(root, "img_dir", "train")
val_image_path = os.path.join(root, "img_dir", "val")

train_images = sorted([os.path.join(train_image_path, item) for item in os.listdir(train_image_path)])[:90]
test_images = sorted([os.path.join(train_image_path, item) for item in os.listdir(train_image_path)])[90:]
val_images = sorted([os.path.join(val_image_path, item) for item in os.listdir(val_image_path)])


train_index = [os.path.join(train_mask_path,get_index(image)) for image in train_images]
test_index = [os.path.join(train_mask_path,get_index(image)) for image in test_images]
val_index = [os.path.join(val_mask_path,get_index(image)) for image in val_images]

train_mask = [os.path.join(train_mask_path, get_mask(image)) for image in train_images]
test_mask = [os.path.join(train_mask_path, get_mask(image)) for image in test_images]
val_mask = [os.path.join(val_mask_path, get_mask(image)) for image in val_images]


folders = ["Train", "Validation", "Test"]
subfolders = ["Annotation__index", "Annotation__color", "Image__8bit_NirRGB"]
for f in folders:
    Path(os.path.join(root, f)).mkdir(parents=True, exist_ok=True)
    for sf in subfolders:
        Path(os.path.join(root, f, sf)).mkdir(parents=True, exist_ok=True)

copy(test_images, os.path.join(root, "Test", "Image__8bit_NirRGB"))
copy(test_index, os.path.join(root, "Test", "Annotation__index"))
copy(test_mask, os.path.join(root, "Test", "Annotation__color"))

copy(val_images, os.path.join(root, "Validation", "Image__8bit_NirRGB"))
copy(val_index, os.path.join(root, "Validation", "Annotation__index"))
copy(val_mask, os.path.join(root, "Validation", "Annotation__color"))

copy(train_images, os.path.join(root, "Train", "Image__8bit_NirRGB"))
copy(train_index, os.path.join(root, "Train", "Annotation__index"))
copy(train_mask, os.path.join(root, "Train", "Annotation__color"))