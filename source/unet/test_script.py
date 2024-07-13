import torch
assert torch.cuda.is_available(), "Notebook is not configured properly!"
device = 'cuda:0'
print("Training network on {}".format(torch.cuda.get_device_name(device=device)))