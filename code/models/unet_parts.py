# Source https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

#!/usr/bin/python

# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
#            U.weight_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1)),
            nn.BatchNorm2d(out_ch),
#            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
#            U.weight_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
            nn.BatchNorm2d(out_ch),
#            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, isize, in_ch, out_ch, nz):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, int(out_ch * 0.5))
        self.project = nn.ConvTranspose2d(nz, int(out_ch * 0.5), isize, 1, 0, bias=False)

    def forward(self, x, z):
        x = self.conv(x)
        noise = self.project(z)
        output = torch.cat([x, noise], dim=1)
        return output


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
