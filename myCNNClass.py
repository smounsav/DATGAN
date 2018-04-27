from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

import models.cnnClass_model as cnnClass_model

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nfilter', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--lrC', type=float, default=0.00005, help='learning rate for Classifier, default=0.00005')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--savedModel', default='', help="path to previously saved model (to continue training)")
parser.add_argument('--outDir', default=None, help='Where to store samples and models')
opt = parser.parse_args()
print(opt)

# define output directory
if opt.outDir is None:
    opt.outDir = 'samples'
os.system('mkdir {0}'.format(opt.outDir))

# start cudnn autotuner
cudnn.benchmark = True

# print warning message if cuda was not explicitely specified
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# define preprocessing
transform=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1

# define datset and initialise dataloader
trainset = dset.ImageFolder(root=opt.dataroot + '/train', transform=transform)
testset = dset.ImageFolder(root=opt.dataroot + '/test', transform=transform)

assert trainset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=opt.workers)
assert testset
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=opt.workers)

# custom weights initialization called on netG and savedModel
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.05)
        m.bias.data.fill_(0)

def weights_init_Linear(m):
    classname = m.__class__.__name__
    if  classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.05)
        m.bias.data.fill_(0)

# initialise classifier
classifier = cnnClass_model.cnnClass(opt.nc, opt.nfilter)
classifier.apply(weights_init)

if opt.cuda:
    classifier.cuda()
# load checkpoint if needed
if opt.savedModel != '':
    pretrained_dict = torch.load(opt.savedModel)
    classifier_dict = classifier.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in classifier_dict}
    # 2. overwrite entries in the existing state dict
    classifier_dict.update(pretrained_dict) 
    # 3. load the new state dict
    classifier.load_state_dict(classifier_dict)
#    classifier.load_state_dict(torch.load(opt.savedModel))
    for param in classifier.features.parameters():
        param.requires_grad = False
print(classifier)

# define loss
criterion = nn.CrossEntropyLoss()
if opt.cuda:
    criterion = criterion.cuda()
# setup optimizer
#lr=opt.lr
#optimizer = optim.SGD([{'params': classifier.layer.parameters()},
#optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=opt.momentum)
if opt.adam:
    optimizerC = optim.Adam(classifier.parameters(), lr=opt.lrC, betas=(opt.beta1, 0.999))
else:

    optimizerC = optim.RMSprop(classifier.parameters(), lr=opt.lrC)

epoch = 0
for epoch in range(opt.niter):
    classifier.train()
    trainloss = 0.0
    nbcorrecttrain = 0
    totaltrain = 0
    for i, data in enumerate(trainloader, 0):
        # Training phase
        # get the inputs
        trainimages, trainlabels = data
        # activate cuda version of the variables if cuda is activated
        if opt.cuda:
            trainimages, trainlabels = trainimages.cuda(), trainlabels.cuda()
        # wrap them in Variable
        trainimages, trainlabels = Variable(trainimages), Variable(trainlabels)
        # forward + backward + optimize
        trainoutput = classifier(trainimages)
        if epoch % 10 == 0 and epoch > 0:
           # get the index of the max log-probability to get the label
            trainpred = trainoutput.data.max(1, keepdim=True)[1]
            # count the number of samples correctly classified
            nbcorrecttrain += trainpred.eq(trainlabels.data.view_as(trainpred)).sum()
            totaltrain += trainlabels.size(0)
            trainacc = 100 * nbcorrecttrain / totaltrain
        # Compute train loss
        trainloss = criterion(trainoutput, trainlabels)
        # zero the parameter gradients
        optimizerC.zero_grad()
        trainloss.backward()
        optimizerC.step()
      
    # print statistics save loss in a log file after each 10 epoch
    if epoch % 10 == 0 and epoch > 0:    # print every 10 epoch

        # Validation phase
        nbcorrectval = 0
        totalval = 0
        classifier.eval()
        for valdata in testloader:
            # Get test images
            valimages, vallabels = valdata
            # activate cuda version of the variables if cuda is activated
            if opt.cuda:
                valimages, vallabels = valimages.cuda(), vallabels.cuda()
            # wrap them in Variable
            valimages, vallabels = Variable(valimages, volatile=True), Variable(vallabels, volatile=True)
            # Calculate scores
            valoutput = classifier(valimages)
            # get the index of the max log-probability to get the label
            valpred = valoutput.data.max(1, keepdim=True)[1]
            # count the number of samples correctly classified
            nbcorrectval += valpred.eq(vallabels.data.view_as(valpred)).sum()
            totalval += vallabels.size(0)
            valacc = 100 * nbcorrectval / totalval
            # Compute val loss
            valloss = criterion(valoutput, vallabels)

        txtLoss = '[%d] Train loss: %.5f Train accuracy: %.2f Test loss: %.5f Test accuracy: %.2f\n' % (epoch, trainloss.data[0], trainacc, valloss.data[0], valacc)
        print(txtLoss)
        f = open(opt.outDir + '/' + opt.outDir + '.txt', 'a')
        f.write(txtLoss)
        f.close

# save final models
torch.save(classifier.state_dict(), '{0}/pyrCNN_epoch_{1}.pth'.format(opt.outDir, epoch + 1))
