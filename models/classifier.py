import torch.nn as nn

class cnnClass(nn.Module):
    def __init__(self, nc, nfilter):
        super(cnnClass, self).__init__()
        self.nfilter = nfilter
        
        features = nn.Sequential()
        # input is nc x isize x isize
        features.add_module('initial.conv.{0}-{1}'.format(nc, nfilter),
                        nn.Conv2d(nc, nfilter, 4, 2, 1, bias=False))
        features.add_module('initial.relu.{0}'.format(nfilter),
                        nn.LeakyReLU(0.2, inplace=True))
        in_feat = nfilter
        out_feat = nfilter * 2                        
        features.add_module('features.{0}-{1}.conv'.format(in_feat, out_feat),
                        nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
#        features.add_module('features.{0}.batchnorm'.format(out_feat),
#                        nn.BatchNorm2d(out_feat))
        features.add_module('features.{0}.relu'.format(out_feat),
                        nn.LeakyReLU(0.2, inplace=True))
        in_feat = nfilter * 2
        out_feat = nfilter * 4
        features.add_module('features.{0}-{1}.conv'.format(in_feat, out_feat),
                        nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
#        features.add_module('features.{0}.batchnorm'.format(out_feat),
#                        nn.BatchNorm2d(out_feat))
        features.add_module('features.{0}.relu'.format(out_feat),
                        nn.LeakyReLU(0.2, inplace=True))
        self.features = features
        fc = nn.Sequential()
        fc.add_module('linear.{0}.{1}'.format(out_feat * 4 * 4, 1024),
                        nn.Linear(out_feat * 4 * 4, 1024))
        fc.add_module('linear.{0}.{1}.dropout'.format(out_feat * 4 * 4, 1024),
                      nn.Dropout(p=0.5, inplace=True))
        fc.add_module('linear.{0}.{1}.relu'.format(out_feat * 4 * 4, 1024),
                        nn.LeakyReLU(0.2, inplace=True))
        fc.add_module('linear.{0}.{1}'.format(1024, 10),
                        nn.Linear(1024, 10))
        fc.add_module('linear.{0}.{1}.dropout'.format(1024, 10),
                      nn.Dropout(p=0.5, inplace=True))
        self.fc = fc
        
    def forward(self, input):
    
        x = self.features.forward(input)
        x = x.view(-1, self.nfilter *4 *4 *4)
        return self.fc.forward(x)