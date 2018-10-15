from __future__ import print_function
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import datetime
from utils import logger as logger

import models.resnet as resnet_model
import models.classifier as classifier_model
import models.wide_resnet as wide_resnet_model
import models.generator as generator_model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | svhn | mnist | stl10 | folder')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--da', action='store_true', help='Whether to apply data augmentation to training dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--trainsetsize', type=int, help='size of training dataset to use, -1 = full dataset', default=-1)
parser.add_argument('--valratio', type=float, default=0.3, help='ratio of the labeled train dataset to be used as validation set')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector for generator')
parser.add_argument('--ncf', type=int, default=64, help='initial number of filters netC')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--optim', default='adam', help='sgd | rmsprop | adam (default is adam)')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--lrC', type=float, default=0.0005, help='learning rate for netC, default=0.0005')


parser.add_argument('--netC', default='', help="path to previously saved model (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--outDirPrefix', default=None, help='Additional text for output path')
parser.add_argument('--outDir', default=None, help='Where to store samples and models')
parser.add_argument('--outDirSuffix', default=None, help='Additional text for output path')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--classModel', required=True, help='badGAN | WRN | ResNet18 | ResNetPA | ShakeShake')
#parser.add_argument('--WRN', action='store_true', help='Use a Wide ResNet as Classifier')
parser.add_argument('--WRNDepth', type=int, default=28, help='Depth factor of the Wide ResNet')
parser.add_argument('--WRNWidth', type=int, default=10, help='Width factor of the Wide ResNet')
parser.add_argument('--WRNDO', type=float, default=0.3, help='DropOut rate of the Wide ResNet')
parser.add_argument('--nostn'  , action='store_true', help='Deactivate STN in generator')
parser.add_argument('--sched', action='store_true', help='Activate LR rate decay scheduler')

opt = parser.parse_args()

assert 0 <= opt.valratio <= 1, 'Error: invalid validation data ratio. valratio should be 0 <= valratio <= 1'

if opt.outDirPrefix is not None:
    outDir = str(opt.outDirPrefix)
else:
    outDir = ''

if opt.outDir is None:
    # now = datetime.datetime.now().timetuple()
    # opt.outDir = 'run-' + str(now.tm_year) + str(now.tm_mon) + str(now.tm_mday) + str(now.tm_hour) + str(now.tm_min) + str(now.tm_sec)
    outDir = outDir + str(opt.dataset).upper() + 'baseline' + 'N' + str(opt.trainsetsize) + 'M' + str(opt.classModel) + 'C' + str(opt.lrC).replace('.', '')
if opt.nostn:
    outDir = outDir + 'NOSTN'
else:
    outDir = outDir + 'STN'
if opt.da:
    outDir = outDir + 'DA'
if opt.sched:
    outDir = outDir + 'SCHED'

if opt.outDirSuffix is not None:
    # now = datetime.datetime.now().timetuple()
    # opt.outDir = 'run-' + str(now.tm_year) + str(now.tm_mon) + str(now.tm_mday) + str(now.tm_hour) + str(now.tm_min) + str(now.tm_sec)
    outDir = outDir + str(opt.outDirSuffix)
print(outDir)
os.system('mkdir {0}'.format(outDir))

# log command line parameters
print(opt)
logger(outDir, 'parameters.txt', str(opt))

# start cudnn autotuner
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# activate cuda acceleration if available
if opt.cuda:
    device = torch.device("cuda")
    pinned_memory = True
else:
    device = torch.device("cpu")
    pinned_memory = False

# define preprocessing
transformTrain=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1

transformTrainDA=transforms.Compose([
   transforms.RandomCrop(32, padding=4),
   transforms.RandomHorizontalFlip(),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1

transformVal=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1

transformTest=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1

transformTrainMNIST=transforms.Compose([
   transforms.Pad(2),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1

transformTrainMNISTDA=transforms.Compose([
   transforms.Pad(2),
   transforms.RandomCrop(32, padding=4),
   transforms.RandomAffine(10, translate=None, scale=(0.5, 2)),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1

transformValMNIST=transforms.Compose([
   transforms.Pad(2),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1

transformTestMNIST=transforms.Compose([
   transforms.Pad(2),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1

transformTrainSTL10=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1

transformTrainSTL10DA=transforms.Compose([
   transforms.RandomCrop(32, padding=4),
   transforms.RandomAffine(10, translate=None, scale=(0.5, 2)),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1

transformValSTL10=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1

transformTestSTL10=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1

transformTrainSVHN=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1

transformTrainSVHNDA=transforms.Compose([
   transforms.RandomCrop(32, padding=4),
   transforms.RandomAffine(10, translate=None, scale=(0.5, 2)),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1

transformValSVHN=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1

transformTestSVHN=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1


transformTrainCIFAR10=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transformTrainCIFAR10DA=transforms.Compose([
   transforms.RandomCrop(32, padding=4),
   transforms.RandomHorizontalFlip(),
   transforms.ToTensor(),
   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transformValCIFAR10=transforms.Compose([
   transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transformTestCIFAR10=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# define datset and initialise dataloader
if opt.dataset in ['folder']:
    # folder dataset
    if opt.da:
        trainset = dset.ImageFolder(root=opt.dataroot + '/train', transform=transformTrainDA)
    else:
        trainset = dset.ImageFolder(root=opt.dataroot + '/train', transform=transformTrain)
    valset = dset.ImageFolder(root=opt.dataroot + '/train', transform=transformVal)
    testset = dset.ImageFolder(root=opt.dataroot + '/test', transform=transformTest)
    nclasses = len(trainset.classes)
elif opt.dataset == 'cifar10':
    if opt.da:
        trainset = dset.CIFAR10(root=opt.dataroot, train=True, download=True, transform=transformTrainCIFAR10DA)
    else:
        trainset = dset.CIFAR10(root=opt.dataroot, train=True, download=True, transform=transformTrainCIFAR10)
    valset = dset.CIFAR10(root=opt.dataroot, train=True, download=True, transform=transformValCIFAR10)
    testset = dset.CIFAR10(root=opt.dataroot, train=False, download=True, transform=transformTestCIFAR10)
    nclasses = 10
elif opt.dataset == 'mnist':
    if opt.da:
        trainset = dset.MNIST(root=opt.dataroot, train=True, download=True, transform=transformTrainMNISTDA)
    else:
        trainset = dset.MNIST(root=opt.dataroot, train=True, download=True, transform=transformTrainMNIST)
    valset = dset.MNIST(root=opt.dataroot, train=True, download=True, transform=transformValMNIST)
    testset = dset.MNIST(root=opt.dataroot, train=False, download=True, transform=transformTestMNIST)
    nclasses = 10
elif opt.dataset == 'svhn':
    if opt.da:
        trainset = dset.SVHN(root=opt.dataroot, split='train', download=True, transform=transformTrainSVHNDA)
    else:
        trainset = dset.SVHN(root=opt.dataroot, split='train', download=True, transform=transformTrainSVHN)
    valset = dset.SVHN(root=opt.dataroot, split='train', download=True, transform=transformValSVHN)
    testset = dset.SVHN(root=opt.dataroot, split='test', download=True, transform=transformTestSVHN)
    nclasses = 10
elif opt.dataset == 'stl10':
    if opt.da:
        trainset = dset.STL10(root=opt.dataroot, split='train', download=True, transform=transformTrainSTL10DA)
    else:
        trainset = dset.STL10(root=opt.dataroot, split='train', download=True, transform=transformTrainSTL10)
    valset = dset.STL10(root=opt.dataroot, split='train', download=True, transform=transformValSTL10)
    testset = dset.STL10(root=opt.dataroot, split='test', download=True, transform=transformTestSTL10)
    nclasses = 10

# Separate training data into fully labeled training and fully labeled validation
len_train = len(trainset)
indices_train = list(range(len_train))
random.shuffle(indices_train)

if opt.trainsetsize != -1:
    assert 0 < opt.trainsetsize <= len_train, 'Error: invalid required training dataset size. Not enough training samples.'
    indices_train = indices_train[:opt.trainsetsize]
len_train_reduced = len(indices_train)
split_train_val = int(opt.valratio * len_train_reduced)
train_idx, val_idx = indices_train[split_train_val:], indices_train[:split_train_val]
print(len(train_idx))
print(len(val_idx))

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)


assert trainset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, sampler=train_sampler,
                                          shuffle=False, num_workers=opt.workers, pin_memory=pinned_memory)
valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batchSize, sampler=val_sampler,
                                          shuffle=False, num_workers=opt.workers, pin_memory=pinned_memory)
assert testset
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=opt.workers, pin_memory=pinned_memory)

# custom weights initialization called on netG and netC
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

if opt.nostn:
    stn = False
else:
    stn = True


# initialise netC
if opt.classModel == 'WRN':
    netC = wide_resnet_model.WideResNet(opt.WRNDepth, nclasses, opt.WRNWidth, opt.WRNDO).to(device)
elif opt.classModel == 'ResNet18':
    netC = resnet_model.ResNet18().to(device)
else:
    netC = classifier_model.badGanClass(opt.imageSize, opt.nc, opt.ncf, opt.ngpu).to(device)
netC.apply(weights_init)
# Generator
netG = generator_model.UNet(opt.imageSize, opt.nc, opt.nc, opt.nz, opt.ngpu, stn).to(device)
netG.apply(weights_init)

# load checkpoint if needed
if opt.netC != '':
    netC.load_state_dict(torch.load(opt.netC))
logger(outDir, 'models.txt', str(netC))

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
logger(outDir, 'models.txt', str(netG))
print('Models loaded\n')

# define loss
criterion = nn.CrossEntropyLoss().to(device)

# setup optimizer
if opt.optim == 'sgd':
    optimizerC = optim.SGD(netC.parameters(), lr=opt.lrC, momentum=0.9, weight_decay=opt.weight_decay)
elif opt.optim == 'rmsprop':
    optimizerC = optim.RMSprop(netC.parameters(), lr=opt.lrC)
else:
    optimizerC = optim.Adam(netC.parameters(), lr=opt.lrC, betas=(opt.beta1, 0.999))

# setup LR scheduler
if opt.sched:
    if opt.dataset == 'cifar10':
        if opt.classModel == 'WRN':
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[60,120,160], gamma=0.2)
        elif opt.classModel == 'ResNet18':
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[150,250,350], gamma=0.1)
        elif opt.classModel == 'badGAN':
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[100,200,300], gamma=0.1)
        else:
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[100,200,300], gamma=0.1)
    elif opt.dataset == 'svhn':
        if opt.classModel == 'WRN':
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[80, 120], gamma=0.1)
        else:
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[100,200,300], gamma=0.1)
    else:
        scheduler = scheduler.MultiStepLR(optimizerC, milestones=[100,200,300], gamma=0.1)

epoch = 0
for epoch in range(opt.niter):
    if opt.sched:
        scheduler.step()
    netC.train()
    trainloss = 0.0
    nbcorrecttrain = 0
    totaltrain = 0
    trainacc = 0.0
    valacc = 0.0
    testacc = 0.0
    for i, (trainimages, trainlabels) in enumerate(trainloader, 0):
        # Training phase
        # activate cuda version of the variables if cuda is activated
        trainimages, trainlabels = trainimages.to(device), trainlabels.to(device)

        # train with generated images
        if opt.netG != '':
            noise = torch.FloatTensor(trainimages.size(0), opt.nz, 1, 1).normal_(0, 0.1).to(device)
            gentrainimages = netG(trainimages, noise)
            # Draw a real between 0 and 1 and take augmented image if result is > 0.5
            draw = torch.rand(trainimages.size(0)).to(device)
            for i in range(trainimages.size(0)):
                if draw[i] > 0.5:
                    trainimages[i] = gentrainimages[i].detach()

        # forward + backward + optimize
        trainoutput = netC(trainimages)
        # Compute train loss
        trainloss = criterion(trainoutput, trainlabels)

        # compute train accuracy
        if epoch % 10 == 0 and epoch > 0:
           # get the index of the max log-probability to get the label
            trainpred = trainoutput.max(1, keepdim=True)[1]
            # count the number of samples correctly classified
            nbcorrecttrain += trainpred.eq(trainlabels.data.view_as(trainpred)).sum()
            totaltrain += trainlabels.size(0)
            trainacc = 100 * nbcorrecttrain.item() / totaltrain
            # Validation phase
            with torch.no_grad():
                nbcorrectval = 0
                totalval = 0
                for valdata in valloader:
                    # Get test images
                    valimages, vallabels = valdata
                    # activate cuda version of the variables if cuda is activated
                    valimages, vallabels = valimages.to(device), vallabels.to(device)
                    # Calculate scores
                    valoutput = netC(valimages)
                    # get the index of the max log-probability to get the label
                    valpred = valoutput.max(1, keepdim=True)[1]
                    # count the number of samples correctly classified
                    nbcorrectval += valpred.eq(vallabels.data.view_as(valpred)).sum()
                    totalval += vallabels.size(0)
                    valacc = 100 * nbcorrectval.item() / totalval
        # zero the parameter gradients
        optimizerC.zero_grad()
        trainloss.backward()
        optimizerC.step()

    # print statistics save loss in a log file after each 10 epoch
    if epoch % 10 == 0 and epoch > 0:    # print every 10 epoch

        # Test phase
        netC.eval()
        with torch.no_grad():
            testloss = 0.0
            nbcorrecttest = 0
            totaltest = 0
            for (testimages, testlabels) in testloader:
                # activate cuda version of the variables if cuda is activated
                testimages, testlabels = testimages.to(device), testlabels.to(device)
                # Calculate scores
                testoutput = netC(testimages)
                # get the index of the max log-probability to get the label
                testpred = testoutput.max(1, keepdim=True)[1]
                # count the number of samples correctly classified
                nbcorrecttest += testpred.eq(testlabels.data.view_as(testpred)).sum()
                totaltest += testlabels.size(0)
                testacc = 100 * nbcorrecttest.item() / totaltest
                # Compute val loss
                testloss = criterion(testoutput, testlabels)

        txtLoss = '{0} Train loss: {1} Train accuracy: {2} Val accuracy {3} Test loss: {4} Test accuracy: {5}'.format(
            epoch,
            trainloss.item(), trainacc,
            valacc,
            testloss.item(), testacc)
        print(txtLoss)
        logger(outDir, 'loss.txt', str(txtLoss))

# save final models
torch.save(netC.state_dict(), '{0}/netC_epoch_{1}.pth'.format(opt.outDir, epoch))
