import torch
import torch.nn as nn

class D_Class(nn.Module):
    def __init__(self, isize, nc, ndf, nclass):
        super(D_Class, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        inputimage = nn.Sequential()
        # input is nc x isize x isize
        inputimage.add_module('inputimage.conv.{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        inputimage.add_module('inputimage.relu.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        self.inputimage = inputimage

        inputlabel= nn.Sequential()
        # input is 1 x nclass
        inputlabel.add_module('inputlabel.deconv.{0}-{1}'.format(nclass, ndf),
                        nn.ConvTranspose2d(nclass, ndf, 16, 1, 0, bias=False))
        inputlabel.add_module('inputlabel.relu.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        self.inputlabel = inputlabel

        csize, cndf = isize / 2, ndf * 2
        layers = nn.Sequential()
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            layers.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
#            layers.add_module('pyramid.{0}.instancenorm'.format(out_feat),
#                            nn.InstanceNorm2d(out_feat))
            layers.add_module('pyramid.{0}.relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        layers.add_module('pyramid.{0}.{1}.dropout'.format(out_feat, out_feat),
                      nn.Dropout2d(p=0.5, inplace=True))
        self.layers = layers

        # state size. K x 4 x 4
        final = nn.Sequential()
        final.add_module('final.{0}-{1}.conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.final = final

    def forward(self, images, labels):
        inputim = self.inputimage(images)
        inputlbl = self.inputlabel(labels.unsqueeze(-1).unsqueeze(-1))
        output = torch.cat([inputim, inputlbl], 1)
        output = self.layers(output)
        output = self.final(output)
        return output.view(-1)


class D_Dist(nn.Module):
    def __init__(self, isize, nc, ndf):
        super(D_Dist, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        inputimage = nn.Sequential()
        # input is nc x isize x isize
        inputimage.add_module('inputimage.conv.{0}-{1}'.format(nc, ndf),
                             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        inputimage.add_module('inputimage.relu.{0}'.format(ndf),
                             nn.LeakyReLU(0.2, inplace=True))
        self.inputimage = inputimage

        layers = nn.Sequential()
        csize, cndf = isize / 2, ndf * 2
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            layers.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                                nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
#            layers.add_module('pyramid.{0}.batchnorm'.format(out_feat),
#                                nn.InstanceNorm2d(out_feat))
            layers.add_module('pyramid.{0}.relu'.format(out_feat),
                                nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2
        layers.add_module('pyramid.{0}.{1}.dropout'.format(out_feat, out_feat),
                      nn.Dropout2d
                      (p=0.5, inplace=True))
        self.layers = layers

        # state size. K x 4 x 4
        final = nn.Sequential()
        final.add_module('final.{0}-{1}.conv'.format(cndf, 1),
                         nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.final = final

    def forward(self, image1, image2):
        inputim1 = self.inputimage(image1)
        inputim2 = self.inputimage(image2)
        input = torch.cat([inputim1, inputim2], 1)
        output = self.layers(input)
        output = self.final(output)
        return output.view(-1)

