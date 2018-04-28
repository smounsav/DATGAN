#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

# python 3 confusing imports :(
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, nz):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels_in, 64, nz)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_channels_out)

    def forward(self, x, z):
        x1 = self.inc(x, z)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
