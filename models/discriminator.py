import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm

class DClass(nn.Module):
    def __init__(self, isize, nc, ndf, nclass):
        super(DClass, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        inputimage = nn.Sequential()
        # input is nc x isize x isize
        inputimage.add_module('inputimage_conv_{0}_{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        inputimage.add_module('inputimage_relu_{0}'.format(ndf),
                        nn.LeakyReLU(0.2))
        self.inputimage = inputimage

        inputlabel= nn.Sequential()
        # input is 1 x nclass
        inputlabel.add_module('inputlabel_deconv_{0}_{1}'.format(nclass, ndf),
                        nn.ConvTranspose2d(nclass, ndf, 16, 1, 0, bias=False))
        inputlabel.add_module('inputlabel_relu_{0}'.format(ndf),
                        nn.LeakyReLU(0.2))
        self.inputlabel = inputlabel

        csize, cndf = isize / 2, ndf * 2
        layers = nn.Sequential()
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            layers.add_module('pyramid_{0}_{1}_conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
#            layers.add_module('pyramid.{0}.instancenorm'.format(out_feat),
#                            nn.InstanceNorm2d(out_feat))
            layers.add_module('pyramid_{0}_relu'.format(out_feat),
                            nn.LeakyReLU(0.2))
            cndf = cndf * 2
            csize = csize / 2

        layers.add_module('pyramid_{0}_{1}_dropout'.format(out_feat, out_feat),
                      nn.Dropout2d(p=0.5))
        self.layers = layers

        # state size. K x 4 x 4
        final = nn.Sequential()
        final.add_module('final_{0}_{1}_conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.final = final

    def forward(self, images, labels):
        inputim = self.inputimage(images)
        inputlbl = self.inputlabel(labels.unsqueeze(-1).unsqueeze(-1))
        output = torch.cat([inputim, inputlbl], 1)
        output = self.layers(output)
        output = self.final(output)
        return output.view(-1)


class DDist(nn.Module):
    def __init__(self, isize, nc, ndf):
        super(DDist, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        inputimage = nn.Sequential()
        # input is nc x isize x isize
        inputimage.add_module('inputimage_conv_{0}_{1}'.format(nc, ndf),
                             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        inputimage.add_module('inputimage_relu_{0}'.format(ndf),
                             nn.LeakyReLU(0.2))
        self.inputimage = inputimage

        layers = nn.Sequential()
        csize, cndf = isize / 2, ndf * 2
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            layers.add_module('pyramid_{0}_{1}_conv'.format(in_feat, out_feat),
                                nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
#            layers.add_module('pyramid.{0}.batchnorm'.format(out_feat),
#                                nn.InstanceNorm2d(out_feat))
            layers.add_module('pyramid_{0}_relu'.format(out_feat),
                                nn.LeakyReLU(0.2))
            cndf = cndf * 2
            csize = csize / 2
        layers.add_module('pyramid_{0}_{1}_dropout'.format(out_feat, out_feat),
                      nn.Dropout2d(p=0.5))
        self.layers = layers

        # state size. K x 4 x 4
        final = nn.Sequential()
        final.add_module('final_{0}_{1}_conv'.format(cndf, 1),
                         nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.final = final

    def forward(self, image1, image2):
        inputim1 = self.inputimage(image1)
        inputim2 = self.inputimage(image2)
        inputcat = torch.cat([inputim1, inputim2], 1)
        output = self.layers(inputcat)
        output = self.final(output)
        return output.view(-1)

class badGanDClass(nn.Module):
    def __init__(self, isize, nc, nfilter, nclass):
        super(badGanDClass, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.nfilter = nfilter

        n_filter_0 = int(nfilter * 0.75)
        n_filter_1 = int(nfilter * 1.5)
        n_filter_2 = nfilter * 3

        inputimage = nn.Sequential()
        # input is nc x isize x isize
        inputimage.add_module('inputimage_conv_{0}_{1}'.format(nc, n_filter_0),
                        nn.Conv2d(nc, n_filter_0, 3, 1, 1, bias=False))
        inputimage.add_module('inputimage_relu_{0}'.format(n_filter_0),
                        nn.LeakyReLU(0.2))
        self.inputimage = inputimage

        inputlabel= nn.Sequential()
        # input is 1 x nclass
        inputlabel.add_module('inputlabel_deconv_{0}_{1}'.format(nclass, n_filter_0),
                        nn.ConvTranspose2d(nclass, n_filter_0, 32, 1, 0, bias=False))
        inputlabel.add_module('inputlabel_relu_{0}'.format(n_filter_0),
                        nn.LeakyReLU(0.2))
        self.inputlabel = inputlabel

        features = nn.Sequential()
        # input is nc x isize x isize
#        features.add_module('initial1_0_conv_{0}_{1}'.format(nc, n_filter_1),
#                            weight_norm(nn.Conv2d(nc, n_filter_1, 3, 1, 1, bias=False)))
#        features.add_module('initial1_0_relu_{0}'.format(n_filter_1),
#                            nn.LeakyReLU(0.2))
        features.add_module('initial1_1_conv_{0}_{1}'.format(n_filter_1, n_filter_1),
                            weight_norm(nn.Conv2d(n_filter_1, n_filter_1, 3, 1, 1, bias=False)))
        features.add_module('initial1_1_relu_{0}'.format(n_filter_1),
                            nn.LeakyReLU(0.2))
        features.add_module('initial1_2_conv_{0}_{1}'.format(n_filter_1, n_filter_1),
                            weight_norm(nn.Conv2d(n_filter_1, n_filter_1, 3, 2, 1, bias=False)))
        features.add_module('initial1_2_relu_{0}'.format(n_filter_1),
                            nn.LeakyReLU(0.2))
        features.add_module('initial1_dropout',
                      nn.Dropout(p=0.5))

        features.add_module('initial2_0_conv_{0}_{1}'.format(n_filter_1, n_filter_2),
                            weight_norm(nn.Conv2d(n_filter_1, n_filter_2, 3, 1, 1, bias=False)))
        features.add_module('initial2_0_relu_{0}'.format(n_filter_2),
                            nn.LeakyReLU(0.2))
        features.add_module('initial2_1_conv_{0}_{1}'.format(n_filter_2, n_filter_2),
                            weight_norm(nn.Conv2d(n_filter_2, n_filter_2, 3, 1, 1, bias=False)))
        features.add_module('initial2_1_relu_{0}'.format(n_filter_2),
                            nn.LeakyReLU(0.2))
        features.add_module('initial2_2_conv_{0}_{1}'.format(n_filter_2, n_filter_2),
                            weight_norm(nn.Conv2d(n_filter_2, n_filter_2, 3, 2, 1, bias=False)))
        features.add_module('initial2_2_relu_{0}'.format(n_filter_2),
                            nn.LeakyReLU(0.2))
        features.add_module('initial2_dropout',
                      nn.Dropout(p=0.5))

        features.add_module('initial3_0_{0}_{1}_conv'.format(n_filter_2, n_filter_2),
                            weight_norm(nn.Conv2d(n_filter_2, n_filter_2, 3, 1, 0, bias=False)))
        features.add_module('initial3_0_relu_{0}'.format(n_filter_2),
                            nn.LeakyReLU(0.2))
        features.add_module('initial3_1_{0}_{1}_conv'.format(n_filter_2, n_filter_2),
                            weight_norm(nn.Conv2d(n_filter_2, n_filter_2, 1, 1, 0, bias=False)))
        features.add_module('initial3_1_relu_{0}'.format(n_filter_2),
                            nn.LeakyReLU(0.2))
        features.add_module('initial3_2_{0}_{1}_conv'.format(n_filter_2, n_filter_2),
                            weight_norm(nn.Conv2d(n_filter_2, n_filter_2, 1, 1, 0, bias=False)))
        features.add_module('initial3_2_relu_{0}'.format(n_filter_2),
                            nn.LeakyReLU(0.2))
        self.features = features
        fc = nn.Sequential()
        fc.add_module('linear_{0}_{1}'.format(n_filter_2, 1),
                      nn.Linear(n_filter_2, 1))

        self.fc = fc

    def forward(self, images, labels):
        inputim = self.inputimage(images)
        inputlbl = self.inputlabel(labels.unsqueeze(-1).unsqueeze(-1))
        output = torch.cat([inputim, inputlbl], 1)
        x = self.features.forward(output)
        x = x.mean(3).mean(2)
        return self.fc.forward(x).view(-1)

class badGanDDist(nn.Module):
    def __init__(self, isize, nc, nfilter,):
        super(badGanDDist, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.nfilter = nfilter

        n_filter_0 = int(nfilter * 0.75)
        n_filter_1 = int(nfilter * 1.5)
        n_filter_2 = nfilter * 3

        inputimage = nn.Sequential()
        # input is nc x isize x isize
        inputimage.add_module('inputimage_conv_{0}_{1}'.format(nc, n_filter_0),
                              nn.Conv2d(nc, n_filter_0, 3, 1, 1, bias=False))
        inputimage.add_module('inputimage_relu_{0}'.format(n_filter_0),
                              nn.LeakyReLU(0.2))
        self.inputimage = inputimage

        features = nn.Sequential()
        # input is nc x isize x isize
        #        features.add_module('initial1_0_conv_{0}_{1}'.format(nc, n_filter_1),
        #                            weight_norm(nn.Conv2d(nc, n_filter_1, 3, 1, 1, bias=False)))
        #        features.add_module('initial1_0_relu_{0}'.format(n_filter_1),
        #                            nn.LeakyReLU(0.2))
        features.add_module('initial1_1_conv_{0}_{1}'.format(n_filter_1, n_filter_1),
                            weight_norm(nn.Conv2d(n_filter_1, n_filter_1, 3, 1, 1, bias=False)))
        features.add_module('initial1_1_relu_{0}'.format(n_filter_1),
                            nn.LeakyReLU(0.2))
        features.add_module('initial1_2_conv_{0}_{1}'.format(n_filter_1, n_filter_1),
                            weight_norm(nn.Conv2d(n_filter_1, n_filter_1, 3, 2, 1, bias=False)))
        features.add_module('initial1_2_relu_{0}'.format(n_filter_1),
                            nn.LeakyReLU(0.2))
        features.add_module('initial1_dropout',
                            nn.Dropout(p=0.5))

        features.add_module('initial2_0_conv_{0}_{1}'.format(n_filter_1, n_filter_2),
                            weight_norm(nn.Conv2d(n_filter_1, n_filter_2, 3, 1, 1, bias=False)))
        features.add_module('initial2_0_relu_{0}'.format(n_filter_2),
                            nn.LeakyReLU(0.2))
        features.add_module('initial2_1_conv_{0}_{1}'.format(n_filter_2, n_filter_2),
                            weight_norm(nn.Conv2d(n_filter_2, n_filter_2, 3, 1, 1, bias=False)))
        features.add_module('initial2_1_relu_{0}'.format(n_filter_2),
                            nn.LeakyReLU(0.2))
        features.add_module('initial2_2_conv_{0}_{1}'.format(n_filter_2, n_filter_2),
                            weight_norm(nn.Conv2d(n_filter_2, n_filter_2, 3, 2, 1, bias=False)))
        features.add_module('initial2_2_relu_{0}'.format(n_filter_2),
                            nn.LeakyReLU(0.2))
        features.add_module('initial2_dropout',
                            nn.Dropout(p=0.5))

        features.add_module('initial3_0_{0}_{1}_conv'.format(n_filter_2, n_filter_2),
                            weight_norm(nn.Conv2d(n_filter_2, n_filter_2, 3, 1, 0, bias=False)))
        features.add_module('initial3_0_relu_{0}'.format(n_filter_2),
                            nn.LeakyReLU(0.2))
        features.add_module('initial3_1_{0}_{1}_conv'.format(n_filter_2, n_filter_2),
                            weight_norm(nn.Conv2d(n_filter_2, n_filter_2, 1, 1, 0, bias=False)))
        features.add_module('initial3_1_relu_{0}'.format(n_filter_2),
                            nn.LeakyReLU(0.2))
        features.add_module('initial3_2_{0}_{1}_conv'.format(n_filter_2, n_filter_2),
                            weight_norm(nn.Conv2d(n_filter_2, n_filter_2, 1, 1, 0, bias=False)))
        features.add_module('initial3_2_relu_{0}'.format(n_filter_2),
                            nn.LeakyReLU(0.2))
        self.features = features
        fc = nn.Sequential()
        fc.add_module('linear_{0}_{1}'.format(n_filter_2, 1),
                      nn.Linear(n_filter_2, 1))

        self.fc = fc

    def forward(self, image1, image2):
        inputim1 = self.inputimage(image1)
        inputim2 = self.inputimage(image2)
        output = torch.cat([inputim1, inputim2], 1)
        x = self.features.forward(output)
        x = x.mean(3).mean(2)
        return self.fc.forward(x).view(-1)