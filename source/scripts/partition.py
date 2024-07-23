import torch
import os
import shutil
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from pathlib import Path

def get_label(image_name):
    return image_name[:-4]+"_24label.png"
def get_coord(image_name):
    return image_name[:-4]+".rpb"
def get_label_color(image_name):
    return image_name[:-4]+"_24label.tif"

def create_partitions(source_folders, dest_folder, data):
    # source folders are always the same
    image_source, index_source, mask_source, coord_source = source_folders
    for partition, folders in dest_folder.items():
        print("Creating {} partition...".format(partition))
        for folder in folders:
            Path(folder).mkdir(parents=True, exist_ok=True)
        image_dest, index_dest, mask_dest, coord_dest = folders
        for image, index, color, coord in data[partition]:
            shutil.move(os.path.join(image_source, image), os.path.join(image_dest, image))
            shutil.move(os.path.join(index_source, index), os.path.join(index_dest, index))
            shutil.move(os.path.join(mask_source, color), os.path.join(mask_dest, color))
            shutil.move(os.path.join(coord_source, coord), os.path.join(coord_dest, coord))

root = "D:\\Datasets\\GID15"

print("Warning: because of storage purposes, the original dataset will be deleted after the partition.")
print("Make sure that you have a backup of your dataset.")
c = input("Press any key to continue...")
ann_index = "Annotation__index"
ann_color = "Annotation__color"
coord_files = "Coordinate_files"
images = "Image__8bit_NirRGB"
image_files = [item for item in os.listdir(os.path.join(root, images))]

label_index_files = [item for item in os.listdir(os.path.join(root, ann_index))]
for f in image_files:
    if not get_label(f) in label_index_files:
        print("Something went wrong")
        raise Exception
random_seed = 42
test_split = .1
validation_split = .1
tot_images = len(image_files)
indices = list(range(tot_images))
np.random.seed(random_seed)
np.random.shuffle(indices)
test_count = int(np.floor(tot_images*test_split))
validation_count = int(np.floor(tot_images*validation_split))

validation_indices = indices[:validation_count]
test_indices = indices[validation_count:validation_count+test_count]
train_indices = indices[validation_count+test_count:]
print(validation_indices)
print(test_indices)
print(train_indices)

data = {
    'Validation': [(image_files[idx], get_label(image_files[idx]), get_label_color(image_files[idx]), get_coord(image_files[idx]))  for idx in validation_indices],
    'Test': [(image_files[idx], get_label(image_files[idx]), get_label_color(image_files[idx]), get_coord(image_files[idx])) for idx in test_indices],
    'Train':[(image_files[idx], get_label(image_files[idx]), get_label_color(image_files[idx]), get_coord(image_files[idx])) for idx in train_indices]
}

source_folders = [os.path.join(root, folder) for folder in [images, ann_index, ann_color, coord_files]]
dest_folders = {partition:[os.path.join(root, partition, folder) for folder in [images, ann_index, ann_color, coord_files]] for partition in ["Validation", "Test", "Train"]}
create_partitions(source_folders, dest_folders, data)