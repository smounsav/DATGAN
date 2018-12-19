import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm

class badGanDC(nn.Module):
    def __init__(self, isize, nc, nfilter, nclass):
        super(badGanDC, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.nfilter = nfilter

        n_filter_0 = int(nfilter * 0.75)
        n_filter_1 = int(nfilter * 1.5)
        n_filter_2 = nfilter * 3

        inputimage = nn.Sequential()
        # input is nc x isize x isize
        inputimage.add_module('inputimage_conv_{0}_{1}'.format(nc, n_filter_0),
                        nn.Conv2d(nc, n_filter_0, 3, int(isize / 32), 1, bias=False))
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
        # features.add_module('initial3_dropout',
        #               nn.Dropout(p=0.5))
        self.features = features
        fc = nn.Sequential()
        fc.add_module('linear_{0}_{1}'.format(n_filter_2, 1),
                      nn.Linear(n_filter_2, 1))

        self.fc = fc

    def forward(self, images, labels):
        inputim = self.inputimage(images)
        inputlbl = self.inputlabel(labels.unsqueeze(-1).unsqueeze(-1))
        output = torch.cat([inputim, inputlbl], 1)
        x = self.features(output)
        x = x.mean(3).mean(2)
        return self.fc.forward(x).view(-1)

class badGanDD(nn.Module):
    def __init__(self, isize, nc, nfilter):
        super(badGanDD, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.nfilter = nfilter

        n_filter_0 = int(nfilter * 0.75)
        n_filter_1 = int(nfilter * 1.5)
        n_filter_2 = nfilter * 3

        inputimage = nn.Sequential()
        # input is nc x isize x isize
        inputimage.add_module('inputimage_conv_{0}_{1}'.format(nc, n_filter_0),
                              nn.Conv2d(nc, n_filter_0, 3, int(isize / 32), 1, bias=False))
        inputimage.add_module('inputimage_relu_{0}'.format(n_filter_0),
                              nn.LeakyReLU(0.2))
        self.inputimage = inputimage

        features = nn.Sequential()
        # input is nc x isize x isize
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
        x = self.features(output)
        x = x.mean(3).mean(2)
        return self.fc.forward(x).view(-1)
