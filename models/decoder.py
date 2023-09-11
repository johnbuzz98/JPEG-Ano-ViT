import torch
import torch.nn as nn
import torch.nn.functional as F

import dct_manip as dm
from utils import rgb_to_dct_ycbcr


class Decoder(nn.Module):
    def __init__(self, image_size: int, patch_size: int, emb_size, domain: str):
        super(Decoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.domain = domain

        self.dec_block1 = nn.Sequential(
            nn.ConvTranspose2d(self.emb_size, 256, (3, 3)),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )

        self.dec_block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (3, 3)),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )

        self.dec_block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (3, 3)),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )

        self.dec_block4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True)
        )

        self.dec_block5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, (3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(True)
        )

        self.up = nn.UpsamplingBilinear2d((image_size, image_size)) # fixed output size

        self.output = nn.Conv2d(16, 3, (3,3), stride=1, padding=1)

    def forward(self,x):
        out = x.transpose(1,2)
        out = out.reshape(x.shape[0], -1, self.image_size//self.patch_size, self.image_size//self.patch_size)
        out = self.dec_block1(out)
        out = self.dec_block2(out)
        out = self.dec_block3(out)
        out = self.dec_block4(out)
        out = self.dec_block5(out)
        out = self.up(out)
        out = self.output(out)
        out = nn.functional.sigmoid(out)
        if self.domain == 'DCT':
            out = rgb_to_dct_ycbcr(out)
        return out