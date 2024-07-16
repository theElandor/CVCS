import torch
from utils import eval_model
import torchvision.transforms as v2
from dataset import PostDamDataset
from nets import Tunet
from torch.utils.data import ConcatDataset
#def eval_model(net, validation_loader, validation_len, device, dataset, show_progress = False, write_output=False, prefix=""):

# START Control variables---------------------------------
load_checkpoint = "/homes/mlugli/checkpoints/tunet1/checkpoint40"
images_path = "/work/cvcs2024/MSseg/Postdam_300x300_full/Images/"
labels_path = "/work/cvcs2024/MSseg/Postdam_300x300_full/Labels"
extension = ".png"
# END Control variables---------------------------------


transforms = v2.Compose([
    v2.GaussianBlur(kernel_size=(15), sigma=5),
    v2.ElasticTransform(alpha=200.0)
])

base_dataset = PostDamDataset(images_path, labels_path, extension)
augmented_dataset = PostDamDataset(images_path, labels_path,extension, transforms=transforms)
dataset = ConcatDataset([base_dataset, augmented_dataset])

# NETWORK INITIALIZATION
#assert torch.cuda.is_available(), "Notebook is not configured properly!"
#device = 'cuda:0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("Warning: GPU is not being used for evaluation.")
net = Tunet(768, 12, 12).to(device)
checkpoint = torch.load(load_checkpoint)
net.load_state_dict(checkpoint['model_state_dict'])
print("Successfully loaded checkpoint /homes/mlugli/checkpoints/tunet1/checkpoint40")


#Dataset train/validation split according to validation split and seed.
batch_size = 4
validation_split = .2
random_seed= 42

dataset_size = len(dataset)
base_indices = list(range(dataset_size//2))
np.random.seed(random_seed)
np.random.shuffle(base_indices)
augmented_indices = [i+(len(dataset)//2) for i in base_indices] # take coresponding augmented images
split = int(np.floor((1-validation_split) * (dataset_size//2)))

val_base_indices = base_indices[split:]
val_noisy_indices = augmented_indices[split:]

valid_base_sampler = SubsetRandomSampler(val_base_indices)
valid_noisy_sampler = SubsetRandomSampler(val_noisy_indices)
#for validation loader batch size is default, so 1.
validation_base_loader = torch.utils.data.DataLoader(dataset ,sampler=valid_base_sampler)
validation_noisy_loader = torch.utils.data.DataLoader(dataset ,sampler=valid_noisy_sampler)
macro, weighted = eval_model(net, validation_base_loader, len(valid_base_indices), device, dataset, show_progress=True)