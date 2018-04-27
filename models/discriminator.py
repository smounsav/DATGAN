import torch
import torch.nn as nn

class D_Class(nn.Module):
    def __init__(self, isize, nc, ndf, nclass):
        super(D_Class, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        convImage = nn.Sequential()
        # input is nc x isize x isize
        convImage.add_module('inputimage.conv.{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        convImage.add_module('inputimage.relu.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        self.convImage = convImage

        convLabel= nn.Sequential()
        # input is 1 x nclass
        convLabel.add_module('inputimage.deconv.{0}-{1}'.format(nclass, ndf),
                        nn.ConvTranspose2d(nclass, ndf, 16, 1, 0, bias=False))
        convLabel.add_module('inputimage.relu.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        self.convLabel = convLabel

        csize, cndf = isize / 2, ndf * 2
        features = nn.Sequential()
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            features.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
#            features.add_module('pyramid.{0}.instancenorm'.format(out_feat),
#                            nn.InstanceNorm2d(out_feat))
            features.add_module('pyramid.{0}.relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        features.add_module('pyramid.{0}.{1}.dropout'.format(out_feat, out_feat),
                      nn.Dropout2d(p=0.2, inplace=True))
        self.features = features

        # state size. K x 4 x 4
        final = nn.Sequential()
        final.add_module('final.{0}-{1}.conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        #final.add_module('final.{0}.{1}.dropout'.format(cndf * 4 * 4, 1),
#                      nn.Dropout(p=0.5, inplace=True))

#        final.add_module('final sigmoid',
#                         nn.Sigmoid())
        self.final = final

    def forward(self, images, labels):
        outputim = self.convImage(images)
        outputlbl = self.convLabel(labels.unsqueeze(-1).unsqueeze(-1))
        output = torch.cat([outputim, outputlbl], 1)
        output = self.features(output)
        output = self.final(output)
#        output = output.mean(0)
        return output.view(-1)


class D_Dist(nn.Module):
    def __init__(self, isize, nc, ndf):
        super(D_Dist, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        features = nn.Sequential()
        # input is nc x isize x isize
        features.add_module('inputimage.conv.{0}-{1}'.format(nc, ndf),
                             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        features.add_module('inputimage.relu.{0}'.format(ndf),
                             nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            features.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                                nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
#            features.add_module('pyramid.{0}.batchnorm'.format(out_feat),
#                                nn.InstanceNorm2d(out_feat))
            features.add_module('pyramid.{0}.relu'.format(out_feat),
                                nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2
        features.add_module('pyramid.{0}.{1}.dropout'.format(out_feat, out_feat),
                      nn.Dropout2d
                      (p=0.2, inplace=True))
        self.features = features

        # state size. K x 4 x 4
        final = nn.Sequential()
        final.add_module('final.{0}-{1}.conv'.format(cndf * 2, 1),
                         nn.Conv2d(cndf * 2, 1, 4, 1, 0, bias=False))
#        final.add_module('final.{0}.{1}.dropout'.format(cndf * 2 , 1),
#                      nn.Dropout(p=0.5  , inplace=True))

        #        final.add_module('final sigmoid',
        #                         nn.Sigmoid())
        self.final = final

    def forward(self, image1, image2):
        outputim1 = self.features(image1)
        outputim2 = self.features(image2)
        output = torch.cat([outputim1, outputim2], 1)
        output = self.final(output)
        #        output = output.mean(0)
        return output.view(-1)

