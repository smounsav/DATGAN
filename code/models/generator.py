#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

# python 3 confusing imports :(
from .unet_parts import *
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm

class UNet(nn.Module):
    def __init__(self, isize, n_channels_in, n_channels_out, nz, stn):
        super(UNet, self).__init__()
        self.stn = stn
        self.inc = inconv(isize, n_channels_in, 64, nz)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_channels_out)

        if stn:
            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = nn.Sequential(
                nn.Linear(1024, 32),
                nn.ReLU(),
                nn.Linear(32, 3 * 2)
            )

            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x, z):
        x1 = self.inc(x, z.unsqueeze(-1).unsqueeze(-1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.stn:
            xs = x5
            xs = xs.mean(3).mean(2)
            xs = xs.view(-1, 1024)
            theta = self.fc_loc(xs)
            theta = theta.view(-1, 2, 3)
            grid = F.affine_grid(theta, x.size())
            xstn = F.grid_sample(x, grid)
            x1stn = self.inc(xstn, z.unsqueeze(-1).unsqueeze(-1))
            x2stn = self.down1(x1stn)
            x3stn = self.down2(x2stn)
            x4stn = self.down3(x3stn)
            x5stn = self.down4(x4stn)
            x = self.up1(x5stn, x4stn)
            x = self.up2(x, x3stn)
            x = self.up3(x, x2stn)
            x = self.up4(x, x1stn)
            x = self.outc(x)
        else:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            x = self.outc(x)
        return x

class STN(nn.Module):
    def __init__(self, isize, n_channels_in,  nz):
        super(STN, self).__init__()
        self.inc = inconv(isize, n_channels_in, 64, nz)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(1024, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x, z):
        x1 = self.inc(x, z.unsqueeze(-1).unsqueeze(-1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = x5.mean(3).mean(2)
        x5 = x5.view(-1, 1024)
        theta = self.fc_loc(x5)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        xstn = F.grid_sample(x, grid)
        return xstn


class SCTG(nn.Module):
    def __init__(self, isize, nc, nfilter, nz):
        super(SCTG, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.nfilter = nfilter
        n_filter_0 = int(nfilter * 0.75)
        n_filter_1 = int(nfilter * 1.5)
        n_filter_2 = nfilter * 3

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            # input is nc x isize x isize
            nn.Conv2d(nc, n_filter_1, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(n_filter_1, n_filter_1, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(n_filter_1, n_filter_1, 3, 2, 1, bias=False)),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            weight_norm(nn.Conv2d(n_filter_1, n_filter_2, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(n_filter_2, n_filter_2, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(n_filter_2, n_filter_2, 3, 2, 1, bias=False)),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            weight_norm(nn.Conv2d(n_filter_2, n_filter_2, 3, 1, 0, bias=False)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(n_filter_2, n_filter_2, 1, 1, 0, bias=False)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(n_filter_2, n_filter_2, 1, 1, 0, bias=False)),
            nn.LeakyReLU(0.2)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(192, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.mean(3).mean(2)
        xs = xs.view(-1, 192)
        #xs = torch.cat([xs, noise.squeeze(-1).squeeze(-1)], dim=1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x

class badGanGen(nn.Module):
    def __init__(self, isize, nc, nz, nclass):
        super(badGanGen, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # input 1 is noise of size nz
        self.inputnoise = nn.Sequential(
#            nn.ConvTranspose2d(nz, 512, 2, 1 ,0, bias=False),
#            nn.LeakyReLU(0.2))
            nn.Linear(nz, 512 * 2 * 2),
            nn.BatchNorm1d(512 * 2 * 2),
            nn.ReLU())
        # input 2 is the label in one hot encoding form
        self.inputlabel= nn.Sequential(
#            nn.ConvTranspose2d(nclass, 512, 2, 1 ,0, bias=False),
#            nn.LeakyReLU(0.2))
            nn.Linear(nclass, 512 * 2 * 2),
            nn.BatchNorm1d(512 * 2 * 2),
            nn.ReLU())
        # input is 1024 x 2 x 2
        self.features = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            weight_norm(nn.ConvTranspose2d(128, nc, 5, 1, 2, 0)),
            nn.Tanh())

    def forward(self, labels, noise):
#        inputnoi = self.inputnoise(noise.unsqueeze(-1).unsqueeze(-1))
        inputnoi = self.inputnoise(noise)
        inputnoi = inputnoi.view(noise.size(0), 512, 2, 2)
#        inputlbl = self.inputlabel(labels.unsqueeze(-1).unsqueeze(-1))
        inputlbl = self.inputlabel(labels)
        inputlbl = inputlbl.view(labels.size(0), 512, 2, 2)
        output = torch.cat([inputnoi, inputlbl], 1)
        x = self.features(output)
        return x
