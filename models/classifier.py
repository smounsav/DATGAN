import torch.nn as nn
import torch.nn.parallel
import torch.nn.utils.weight_norm as weight_norm

class cnnClass(nn.Module):
    def __init__(self, nc, nfilter, ngpu):
        super(cnnClass, self).__init__()
        self.ngpu = ngpu
        self.nfilter = nfilter
        
        features = nn.Sequential()
        # input is nc x isize x isize
        features.add_module('initial_conv_{0}_{1}'.format(nc, nfilter),
                        nn.Conv2d(nc, nfilter, 4, 2, 1, bias=False))
        features.add_module('initial_relu_{0}'.format(nfilter),
                        nn.LeakyReLU(0.2))
        in_feat = nfilter
        out_feat = nfilter * 2                        
        features.add_module('features_{0}_{1}_conv'.format(in_feat, out_feat),
                        nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
#        features.add_module('features.{0}.batchnorm'.format(out_feat),
#                        nn.BatchNorm2d(out_feat))
        features.add_module('features_{0}_relu'.format(out_feat),
                        nn.LeakyReLU(0.2))
        in_feat = nfilter * 2
        out_feat = nfilter * 4
        features.add_module('features_{0}_{1}_conv'.format(in_feat, out_feat),
                        nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
#        features.add_module('features.{0}.batchnorm'.format(out_feat),
#                        nn.BatchNorm2d(out_feat))
        features.add_module('features_{0}_relu'.format(out_feat),
                        nn.LeakyReLU(0.2))
        self.features = features
        fc = nn.Sequential()
        fc.add_module('linear_{0}_{1}'.format(out_feat * 4 * 4, 1024),
                        nn.Linear(out_feat * 4 * 4, 1024))
        fc.add_module('linear_{0}_{1}_dropout'.format(out_feat * 4 * 4, 1024),
                      nn.Dropout(p=0.5))
        fc.add_module('linear_{0}_{1}_relu'.format(out_feat * 4 * 4, 1024),
                        nn.LeakyReLU(0.2))
        fc.add_module('linear_{0}_{1}'.format(1024, 10),
                        nn.Linear(1024, 10))
        fc.add_module('linear_{0}_{1}_dropout'.format(1024, 10),
                      nn.Dropout(p=0.5))
        self.fc = fc
        
    def forward(self, input):
        if isinstance(input, torch.cuda.FloatTensor) and self.ngpu > 1:
            x = nn.parallel.data_parallel(self.features, input, range(self.ngpu))
        else:
            x = self.features(input)
        x = x.view(-1, self.nfilter *4 *4 *4)
        return self.fc.forward(x)


class badGanClass(nn.Module):
    def __init__(self, isize, nc, nfilter, ngpu):
        super(badGanClass, self).__init__()
        self.ngpu = ngpu
        self.nfilter = nfilter

        n_filter_1 = int(nfilter * 1.5)
        n_filter_2 = nfilter * 3

        features = nn.Sequential()
        # input is nc x isize x isize
        features.add_module('initial1_0_conv_{0}_{1}'.format(nc, n_filter_1),
                            weight_norm(nn.Conv2d(nc, n_filter_1, 3, int(isize / 32), 1, bias=False)))
        features.add_module('initial1_0_relu_{0}'.format(n_filter_1),
                            nn.LeakyReLU(0.2))
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
        fc.add_module('linear_{0}_{1}'.format(n_filter_2, 10),
                      nn.Linear(n_filter_2, 10))

        self.fc = fc

    def forward(self, input):
        if isinstance(input, torch.cuda.FloatTensor) and self.ngpu > 1:
            x = nn.parallel.data_parallel(self.features, input, range(self.ngpu))
        else:
            x = self.features(input)
        x = x.mean(3).mean(2)
        return self.fc.forward(x)