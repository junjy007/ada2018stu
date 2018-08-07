import functools
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from .resnet import ResnetGenerator

image_transform = transforms.Compose([
            transforms.Resize([256, 256], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])

def load_image(image_path):
    A_img = Image.open(image_path).convert('RGB')
    return image_transform(A_img).unsqueeze(0)

def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
           (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        __patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

def load_generator_from(model_path):
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False,
                                   track_running_stats=True)
    netG = ResnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=False,
                           n_blocks=9)
    state_dict = torch.load(model_path)
    for key in list(
            state_dict.keys()):  # need to copy keys here because we mutate in loop
        __patch_instance_norm_state_dict(state_dict, netG, key.split('.'))
    netG.load_state_dict(state_dict)
    return netG

