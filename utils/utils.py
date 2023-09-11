import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torch_dct as td


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

def rgb_to_ycbcr_torch(img):
    # Check if input is a torch.Tensor
    if not torch.is_tensor(img):
        raise ValueError("Expected input to be a torch.Tensor, but got {}".format(type(img)))

    # Define transformation matrix from RGB to YCbCr
    M = torch.tensor([[0.299, 0.587, 0.114],
                      [-0.1687, -0.3313, 0.5],
                      [0.5, -0.4187, -0.0813]], dtype=img.dtype, device=img.device)

    # Convert using einsum
    ycbcr = torch.einsum('bchw,ch->bchw', img, M)
    ycbcr[:, 1:3, :, :] += 128  # Adjust Cb and Cr components

    return ycbcr

def apply_dct_to_patches(channel):
    # Reshape the tensor to consider 8x8 patches
    b, h, w = channel.shape
    reshaped = channel.view(b, h//8, 8, w//8, 8).permute(0, 1, 3, 2, 4)

    # Apply DCT along the last two dimensions (8x8 patches)
    dct_transformed = td.dct(td.dct(reshaped.transpose(-1, -2), norm='ortho').transpose(-1, -2), norm='ortho')
    dct_transformed = dct_transformed.unsqueeze(1)
    return dct_transformed

def down_sample_C(channel):
    channel_downsampled = F.interpolate(channel.unsqueeze(1), scale_factor=0.5, mode='bilinear', align_corners=False)
    return channel_downsampled.squeeze(1)
def rgb_to_dct_ycbcr(rgb_tensor):
    # Convert to YCbCr
    #ycbcr_tensor = rgb_to_ycbcr_torch(rgb_tensor)
    
    # Apply DCT to 8x8 patches for each channel
    y_dct = apply_dct_to_patches(rgb_tensor[:, 0, :, :])
    cb_dct = apply_dct_to_patches(down_sample_C(rgb_tensor[:, 1, :, :]))
    cr_dct = apply_dct_to_patches(down_sample_C(rgb_tensor[:, 2, :, :]))
    
    return y_dct, torch.cat((cb_dct, cr_dct), dim=1)