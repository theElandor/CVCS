import os
import typing
import torch.utils.data
import torch, torchvision
from pathlib import Path

from torch.utils.data.dataset import T_co
from torchvision.transforms import v2
from torchvision import tv_tensors
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import utils, sys, yaml, transformers
import numpy as np
from transformers import SegformerForSemanticSegmentation

class PatchIterator:
    def __init__(self, patches, chunk_size, load_func):
        self.patches = patches
        self.chunk_size = chunk_size
        self.load_func = load_func
        self.index = 0
        self.total_patches = len(patches)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.total_patches:
            raise StopIteration

        batch_start = (self.index // self.chunk_size) * self.chunk_size
        batch_end = min(batch_start + self.chunk_size, self.total_patches)

        if all(patch is None for patch in self.patches[batch_start:batch_end]):
            self.patches = self.load_func(batch_start, batch_end)

        patch = self.patches[self.index]
        self.index += 1


        return patch


class GID15(torch.utils.data.IterableDataset):
    def __init__(self, parent_dir, patch_shape, chunk_size=1, dict_layout=False, image_transforms=None,
                 geometric_transforms=None):
        super(GID15).__init__()
        assert Path(parent_dir).is_dir(), "Please provide a valid directory."
        self._parent_dir = parent_dir
        self._img_dir = os.path.join(self._parent_dir, 'Image__8bit_NirRGB')
        self._mask_dir = os.path.join(self._parent_dir, 'Annotation__color')
        self._index_dir = os.path.join(self._parent_dir, 'Annotation__index')
        self._files = [Path(item).stem for item in os.listdir(path=self._img_dir)]
        self._files_idxs = torch.arange(self._files.__len__(), dtype=torch.int8)
        self._files_idxs = torch.randperm(self._files_idxs.size()[0])
        self._last_img_loaded = 0
        # self._images_path = [os.path.join(self._img_dir, item + '.tif') for item in self._files]
        # self._masks_path = [os.path.join(self._mask_dir, item + '_15label.tif') for item in self._files]
        # self._indexes_path = [os.path.join(self._index_dir, item + '_15label.png') for item in self._files]
        self._chunk_size = chunk_size
        self._patch_shape = patch_shape
        self._image_shape = (6800, 7200)  # H * W
        self._tpi = (self._image_shape[0] // self._patch_shape) * (
                self._image_shape[1] // self._patch_shape)
        # self.patches = torch.zeros((self.__len__(), self._patch_shape, self._patch_shape), dtype=torch.uint8)
        self._dict_layout = dict_layout
        self._image_transforms = image_transforms
        self._geometric_transforms = geometric_transforms
        if self._dict_layout:
            self._patches = [{} for i in range(self.__len__())]
        else:
            self._patches = [None for i in range(self.__len__())]

        self._win_size = self._tpi * self._chunk_size
        self._load_chunk(0, self._win_size)

    def _load_chunk(self, start, end):
        print("Loading chunk...{}".format(start // self._win_size))
        if self._dict_layout:
            self._patches = [{} for i in range(self.__len__())]
        else:
            self._patches = [None for i in range(self.__len__())]

        _last_chunk_loaded = start // self._win_size
        _files_path = [self._files[i] for i in
                       self._files_idxs[_last_chunk_loaded * self._chunk_size: (
                               _last_chunk_loaded * self._chunk_size + self._chunk_size)]]
        images = [tv_tensors.Image(Image.open(os.path.join(self._img_dir, name + '.tif'))) for name in _files_path]
        index_masks = [tv_tensors.Image(Image.open(os.path.join(self._index_dir, name + '_15label.png'))) for name in
                       _files_path]
        color_masks = [tv_tensors.Image(Image.open(os.path.join(self._mask_dir, name + '_15label.tif'))) for name in
                       _files_path]

        _patches = list()
        _offset_x, _offset_y = 0, 0
        for i in range(self._chunk_size):
            for ii in range(self._tpi):
                _rnd_x = torch.randint(-20, 21, (1,)).item()
                _rnd_y = torch.randint(-20, 21, (1,)).item()

                patch = v2.functional.crop(images[i], np.clip(_offset_y + _rnd_y, 0, 6800),
                                           np.clip(_offset_x + _rnd_x, 0, 7200),
                                           self._patch_shape,
                                           self._patch_shape)
                patch_mask = v2.functional.crop(color_masks[i], np.clip(_offset_y + _rnd_y, 0, 6800),
                                                np.clip(_offset_x + _rnd_x, 0, 7200),
                                                self._patch_shape,
                                                self._patch_shape)
                patch_idxs = v2.functional.crop(index_masks[i], np.clip(_offset_y + _rnd_y, 0, 6800),
                                                np.clip(_offset_x + _rnd_x, 0, 7200),
                                                self._patch_shape,
                                                self._patch_shape)
                if self._image_transforms is not None:
                    patch = self._image_transforms(patch)

                if self._geometric_transforms is not None:
                    concatenation = torch.concat((patch, patch_idxs, patch_mask), dim=0)
                    concatenation = self._geometric_transforms(concatenation)
                    patch = concatenation[:3, :, :]
                    patch_idxs = concatenation[3, :, :]
                    patch_mask = concatenation[4:, :, :]

                if self._dict_layout:
                    _patches.append({'pixel_values': patch.float(), 'labels': patch_idxs.squeeze(0).long(),
                                     'label_ids': patch_mask})
                else:
                    _patches.append((patch, patch_mask, patch_idxs))

                _offset_x += self._patch_shape
                if _offset_x >= self._image_shape[1]:
                    _offset_x = 0
                    _offset_y += self._patch_shape
        self._patches[start:end] = _patches
        self._last_img_loaded += self._chunk_size
        return self._patches

    def __iter__(self):
        return PatchIterator(self._patches, self._chunk_size, self._load_chunk)

    # def __getitem__(self, index):
    #     if self._patches[index] == {} or self._patches[index] == None:
    #         batch_start = (index // self._win_size) * self._win_size
    #         batch_end = min(batch_start + self._win_size, self._tpi * self._files.__len__())
    #
    #         self._load_chunk(batch_start, batch_end)
    #
    #     return self._patches[index]

    def __len__(self):
        return self._tpi * self._files.__len__()


inFile = sys.argv[1]

with open(inFile, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print("LOADED CONFIGURATIONS:")
utils.display_configs(config)
# START Control variables---------------------------------
# This section contains some variables that need to be set before running the script.

checkpoint_directory = config['checkpoint_directory']
img_transforms, geometric_transform = utils.load_basic_transforms(config)
train_data = GID15(config['train'], 224, chunk_size=config['chunk_size'], dict_layout=False,
                   image_transforms=img_transforms, geometric_transforms=geometric_transform)
dl = torch.utils.data.DataLoader(train_data, batch_size=16)

for i in tqdm(dl):
    print(i)

#
# train_args = transformers.TrainingArguments(output_dir=config['logging_dir'], per_device_train_batch_size=16,
#                                             per_gpu_eval_batch_size=1, logging_dir=config['logging_dir'],
#                                             num_train_epochs=2)

