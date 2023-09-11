import os
from glob import glob

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

import dct_manip as dm
import utils.custom_transforms as ctrans


class MvtecAd(Dataset):
    def __init__(
        self, 
        datadir: str, 
        target: str, 
        is_train: bool, 
        resize: int = 224,
        image_domain: str = 'RGB', 
        image_format: str = 'PNG'
    ):
        if image_domain not in ['RGB', 'DCT']:
            raise ValueError(f"Invalid image_domain: {image_domain}. Expected 'RGB' or 'DCT'.")
        
        if image_format not in ['JPEG', 'PNG']:
            raise ValueError(f"Invalid image_format: {image_format}. Expected 'JPEG' or 'PNG'.")

        # load image file list
        self.image_domain = image_domain
        self.image_format = image_format

        if image_format == 'PNG':
            self.datadir = datadir
        else:
            self.datadir = datadir+'_jpeg'
        self.target = target
        self.file_list = glob(os.path.join(self.datadir, self.target, 'train/*/*' if is_train else 'test/*/*'))
        
        # transform ndarray into tensor 
        self.resize = resize
        if self.image_domain == 'RGB':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.resize)
            ])
        else:
            self.transform = transforms.Compose([
                ctrans.Resize_DCT(int(self.resize/8)), # 28 = 224x224, 48 = 384x384
                ctrans.ToRange(val_min=-1, val_max=1, orig_min=-1024, orig_max=1016, dtype=torch.float32),
            ])
        
        
    def __getitem__(self, idx):
        
        file_path = self.file_list[idx]
        
        # image load 
        if self.image_format == 'JPEG':
            _, _, Y, cbcr = dm.read_coefficients(file_path)
            mask_path = file_path.replace('test','ground_truth').replace('.JPEG','_mask.JPEG')
            img = (Y, cbcr)
        else:
            img = Image.open(file_path)
            mask_path = file_path.replace('test','ground_truth').replace('.png','_mask.png')
            if self.image_domain == 'DCT':
                _, _, Y, cbcr = dm.quantize_at_quality(F.pil_to_tensor(img), quality=100)
                img = (Y, cbcr)
            else:
                img = Image.open(file_path)     

        # target
        target = 0 if 'good' in self.file_list[idx] else 1         
        
        # mask
        if 'good' in file_path:
            mask = np.zeros((self.resize, self.resize), dtype=np.float32)
        else:
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((self.resize, self.resize))
            mask = np.array(mask, dtype=bool)
            mask = np.array(mask, dtype=np.float32)

        # convert ndarray into tensor
        img = self.transform(img)
        mask = torch.Tensor(mask).to(torch.int64)
        
        return img, mask, target
        
            
    def __len__(self):
        return len(self.file_list)



def dct_to_rgb_batch(coeff_batch):
    """
    Args:
        coeff_batch (Tensor or tuple): dct coefficient batch with five channels in (b, c, h, w, kh, kw)
                                       if tuple, (Y, cbcr) is expected.

    Returns:
        Tensor: Batch of converted YCbCr to RGB data
    """
    Y_batch, cbcr_batch = coeff_batch
    assert Y_batch.dtype == torch.float32 and cbcr_batch.dtype == torch.float32, f"Y and CbCr dtype should be torch.float32. Current:{Y_batch.dtype}, {cbcr_batch.dtype}"

    B, _, H, W, KH, KW = Y_batch.shape
    _, _, CH, CW, _, _ = cbcr_batch.shape

    # Move tensors to the device Y_batch is on (assumed to be the CUDA device).
    device = Y_batch.device
    dim_inferred = torch.tensor([[H*KH, W*KW], [CH*KH, CW*KW], [CH*KH, CW*KW]], dtype=torch.int32).to(device).expand(B, -1, -1)
    quant_100 = 2 * torch.ones((B, 3, 8, 8), dtype=torch.int16).to(device)

    RGBimg_list = []

    for i in range(B):
        RGBimg = dm.decode_coeff(
            dim_inferred[i], 
            quant_100[i],
            (Y_batch[i]/2).round().to(torch.int16).clamp(min=-1024, max=1016).contiguous(),
            (cbcr_batch[i]/2).round().to(torch.int16).clamp(min=-1024, max=1016).contiguous()
        )
        RGBimg_list.append(RGBimg)

    return torch.stack(RGBimg_list).to(device)
