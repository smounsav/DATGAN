import os
import argparse
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
import json
import glob
from shutil import copyfile

from utils import logger as logger
from cutout import Cutout as Cutout

import models.classifier as classifier_model
import models.wide_resnet as wide_resnet_model
import models.resnet as resnet_model
import models.preact_resnet as preact_resnet_model
import models.shake_resnet as shake_resnet_model
import models.generator as generator_model

parser = argparse.ArgumentParser()
# dataset
parser.add_argument('--dataset', required=True, help='cifar10 | svhn | mnist | fashionmnist | stl10 | folder')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--trainsetsize', type=int, help='size of training dataset to use, -1 = full dataset', default=-1)
parser.add_argument('--valratio', type=float, default=0.3, help='ratio of the labeled train dataset to be used as validation set')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
# dataset preprocessing
parser.add_argument('--da', action='store_true', help='Whether to apply data augmentation to training dataset')
parser.add_argument('--lightda', action='store_true', help='Whether to apply light data augmentation (only crop) to training dataset')
parser.add_argument('--cutout', action='store_true', help='Whether to apply Cutout on training dataset')
parser.add_argument('--cutoutsize', type=int, default=16, help='Cutout size to apply on training dataset')
# model
parser.add_argument('--classModel', default='badGAN', help='badGAN | WRN | ResNet18 | ResNetPA | ShakeShake, default=badGAN')
parser.add_argument('--RNDepth', type=int, default=28, help='Depth factor for Wide ResNet and ShakeShake')
parser.add_argument('--RNWidth', type=int, default=10, help='Width factor for Wide ResNet and ShakeShake')
parser.add_argument('--RNDO', type=float, default=0.3, help='DropOut rate for Wide ResNet')
parser.add_argument('--nostn'  , action='store_true', help='Deactivate STN in generator')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector for generator')
parser.add_argument('--ncf', type=int, default=64, help='initial number of filters netC')
# training
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--optim', default='adam', help='sgd | rmsprop | adam (default is adam)')
parser.add_argument('--lrC', type=float, default=0.0005, help='learning rate for netC, default=0.0005')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer, default=0.9')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay factor, default=0')
parser.add_argument('--sched', action='store_true', help='Activate LR rate decay scheduler')
parser.add_argument('--fixedSeed', type=int, default=None, help='fix seed')
# checkpoints
parser.add_argument('--predParams', default='', help="path to predefined parameters")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netC', default='', help="path to previously saved model (to continue training)")
# system
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--outDirPrefix', default=None, help='Additional text for output path')
parser.add_argument('--outDir', default=None, help='Where to store samples and models')
parser.add_argument('--outDirSuffix', default=None, help='Additional text for output path')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pinnedmemory', action='store_true', help='Whether to use GPU pinned memory')

opt = parser.parse_args()

assert 0 <= opt.valratio <= 1, 'Error: invalid validation data ratio. valratio should be 0 <= valratio <= 1'

# create output directory
if opt.outDirPrefix is not None:
    outDir = str(opt.outDirPrefix)
else:
    outDir = ''
if opt.outDir is None:
    outDir = outDir + str(opt.dataset).upper() + 'baseline' + 'N' + str(opt.trainsetsize) + 'M' + str(opt.classModel) \
            + 'O' + str(opt.optim).upper()
    if opt.optim == 'adam':
                outDir = outDir + 'BT1' + str(opt.beta1).replace('.', '')
    if opt.optim == 'sgd':
        outDir = outDir + 'MOM' + str(opt.momentum).replace('.', '')
    if opt.optim == 'adam' or opt.optim == 'sgd':
        outDir = outDir + 'WD' + str(opt.weight_decay).replace('.', '')
    outDir = outDir + 'B' + str(opt.batchSize) +'C' + str(opt.lrC).replace('.', '')
if opt.netG != '':
    if opt.nostn:
        outDir = outDir + 'NOSTN'
    else:
        outDir = outDir + 'STN'
if opt.lightda:
    outDir = outDir + 'LIDA'
if opt.da:
    outDir = outDir + 'DA'
if opt.cutout:
    outDir = outDir + 'CO'
if opt.sched:
    outDir = outDir + 'SCHED'
if opt.outDirSuffix is not None:
    outDir = outDir + str(opt.outDirSuffix)
print(outDir)
os.system('mkdir {0}'.format(outDir))

currentIteration = 0
job_running = False

# if job is restarted, check if it was interrupted during model saving and use the right model accordingly
newJobRunningFile = Path('{0}/job_running_new'.format(outDir))
newNetG = Path('{0}/netG_running_new'.format(outDir))
newNetC = Path('{0}/netC_running_new'.format(outDir))
if newJobRunningFile.exists() and newNetC.exists() and newNetG.exists():
    copyfile('{0}/job_running_new'.format(outDir), '{0}/job_running'.format(outDir))
    copyfile('{0}/netG_running_new'.format(outDir), '{0}/netG_running'.format(outDir))
    copyfile('{0}/netC_running_new'.format(outDir), '{0}/netC_running'.format(outDir))
    os.remove('{0}/job_running_new'.format(outDir))
    os.remove('{0}/netG_running_new'.format(outDir))
    os.remove('{0}/netC_running_new'.format(outDir))

# if job is restarted after interruption, reload parameters
ongoingFile = Path('{0}/job_running'.format(outDir))
if ongoingFile.exists():
    with open('{0}/job_running'.format(outDir)) as file:
        jobParams = json.loads(file.read())
    job_running =True
    opt.manuelSeed = jobParams['seed']
    currentIteration = jobParams['currentIteration'] + 1
    train_idx = torch.tensor(jobParams['train_idx'])
    val_idx = torch.tensor(jobParams['val_idx'])
    opt.netG = outDir + '/netG_running'
    opt.netC = outDir + '/netC_running'
    print('Restarting job\n')
else:
    # clean directory if already existing
    filelist = glob.glob('{0}/*'.format(outDir))
    for file in filelist:
        os.remove(file)
    # load predefined parameters
    if opt.predParams != '':
        predefinedParams = Path(opt.predParams)
        if predefinedParams.exists():
            with open(predefinedParams) as file:
                jobParams = json.loads(file.read())
            opt.fixedSeed = jobParams['Random seed']
            train_idx = torch.tensor(jobParams['Train dataset'])
            val_idx = torch.tensor(jobParams['Validation dataset'])
            print('Predefined parameters loaded\n')
    # log command line parameters
    print(opt)
    logger(outDir, 'parameters.txt', str(opt))
    print('Starting job\n')

# define and log random seed for reproductibility
if opt.fixedSeed is not None:
    seed = opt.fixedSeed
else:
    seed = random.randint(1, 10000) # fix seed
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
if not job_running:
    print("Random Seed: ", seed)
    logger(outDir, 'parameters.txt', 'Random seed: ' + str(seed)) # log random seed

# change from cudnn autotuner to cudnn deterministic
cudnn.benchmark = False
cudnn.deterministic = True
# activate cuda acceleration if available
if torch.cuda.is_available():
    if opt.cuda:
            device = torch.device("cuda")
            pinned_memory = opt.pinnedmemory
    else:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    if opt.cuda:
        print("WARNING: You have no CUDA device, but you tried to run with --cuda")
    else:
        device = torch.device("cpu")
        pinned_memory = False

# define preprocessing
if opt.dataset == 'svhn':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
elif opt.dataset == 'cifar10':
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
else:
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

transformTrain = transforms.Compose([])
if opt.dataset in ['mnist', 'fashionmnist']:
    transformTrain.transforms.append(transforms.Pad(2))
if opt.lightda:
    if opt.dataset in ['mnist', 'fashionmnist']:
        transformTrain.transforms.append(transforms.RandomCrop(32, padding=4))
    elif opt.dataset in ['svhn']:
        transformTrain.transforms.append(transforms.RandomCrop(32, padding=4))
    elif opt.dataset in ['cifar10']:
        transformTrain.transforms.append(transforms.RandomCrop(32, padding=4))

if opt.da:
    if opt.dataset in ['mnist', 'fashionmnist']:
        transformTrain.transforms.append(transforms.RandomCrop(32, padding=4))
        transformTrain.transforms.append(transforms.RandomAffine(10, translate=None, scale=(0.5, 2)))
    elif opt.dataset in ['svhn']:
        transformTrain.transforms.append(transforms.RandomCrop(32, padding=4))
        transformTrain.transforms.append(transforms.RandomAffine(10, translate=None, scale=(0.5, 2)))
    elif opt.dataset in ['cifar10']:
        transformTrain.transforms.append(transforms.RandomCrop(32, padding=4))
        transformTrain.transforms.append(transforms.RandomHorizontalFlip())
    else:
        transformTrain.transforms.append(transforms.RandomCrop(32, padding=4))
        transformTrain.transforms.append(transforms.RandomHorizontalFlip())
transformTrain.transforms.append(transforms.ToTensor())
transformTrain.transforms.append(normalize)
if opt.cutout:
        transformTrain.transforms.append(Cutout(1, opt.cutoutsize))

transformVal = transforms.Compose([
    transforms.ToTensor(),
    normalize])

transformTest = transforms.Compose([
    transforms.ToTensor(),
    normalize])

# define dataset and initialise dataloader
if opt.dataset in ['folder']:
    # folder dataset
    trainset = dset.ImageFolder(root=opt.dataroot + '/train', transform=transformTrain)
    valset = dset.ImageFolder(root=opt.dataroot + '/train', transform=transformVal)
    testset = dset.ImageFolder(root=opt.dataroot + '/test', transform=transformTest)
    nclasses = len(trainset.classes)
elif opt.dataset == 'cifar10':
    trainset = dset.CIFAR10(root=opt.dataroot, train=True, download=True, transform=transformTrain)
    valset = dset.CIFAR10(root=opt.dataroot, train=True, download=True, transform=transformVal)
    testset = dset.CIFAR10(root=opt.dataroot, train=False, download=True, transform=transformTest)
    nclasses = 10
elif opt.dataset == 'mnist':
    trainset = dset.MNIST(root=opt.dataroot, train=True, download=True, transform=transformTrain)
    valset = dset.MNIST(root=opt.dataroot, train=True, download=True, transform=transformVal)
    testset = dset.MNIST(root=opt.dataroot, train=False, download=True, transform=transformTest)
    nclasses = 10
elif opt.dataset == 'fashionmnist':
    trainset = dset.FashionMNIST(root=opt.dataroot, train=True, download=True, transform=transformTrain)
    valset = dset.FashionMNIST(root=opt.dataroot, train=True, download=True, transform=transformVal)
    testset = dset.FashionMNIST(root=opt.dataroot, train=False, download=True, transform=transformTest)
    nclasses = 10
elif opt.dataset == 'svhn':
    trainset = dset.SVHN(root=opt.dataroot, split='train', download=True, transform=transformTrain)
    valset = dset.SVHN(root=opt.dataroot, split='train', download=True, transform=transformVal)
    testset = dset.SVHN(root=opt.dataroot, split='test', download=True, transform=transformTest)
    nclasses = 10
elif opt.dataset == 'stl10':
    trainset = dset.STL10(root=opt.dataroot, split='train', download=True, transform=transformTrain)
    valset = dset.STL10(root=opt.dataroot, split='train', download=True, transform=transformVal)
    testset = dset.STL10(root=opt.dataroot, split='test', download=True, transform=transformTest)
    nclasses = 10

# Separate training data into training and validation
if not job_running:
    if opt.predParams == '':
        len_train = len(trainset)
    #    indices_train = list(range(len_train))
        shuffled_indices_train = torch.randperm(len_train)
    #    shuffled_indices_train = indices_train
    #    random.shuffle(shuffled_indices_train)
        if opt.trainsetsize != -1:
            assert 0 < opt.trainsetsize <= len_train, 'Error: invalid required training dataset size. Not enough training samples.'
            indices_train_reduced = shuffled_indices_train[:opt.trainsetsize]
        else:
            indices_train_reduced = shuffled_indices_train
        len_train_reduced = len(indices_train_reduced)
        # Training dataset is reduced to [Validation labeled samples:Training labeled samples]
        split_train_val = int(opt.valratio * len_train_reduced)
        train_idx, val_idx = indices_train_reduced[split_train_val:], indices_train_reduced[:split_train_val]
    #    print(len(train_idx))
    #    print(len(val_idx))
    logger(outDir, 'parameters.txt', 'Train dataset: ' + str(train_idx.tolist())) # Save samples used for training
    logger(outDir, 'parameters.txt', 'Validation dataset: ' + str(val_idx.tolist())) # Save samples used for validation
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

# initialise Classifier
if opt.classModel == 'WRN':
    netC = wide_resnet_model.WideResNet(opt.nc, opt.RNDepth, nclasses, opt.RNWidth, opt.RNDO).to(device)
elif opt.classModel == 'ResNet18':
        netC = resnet_model.ResNet18(opt.nc).to(device)
elif opt.classModel == 'ResNetPA':
    netC = preact_resnet_model.PreActResNet18(opt.nc).to(device)
elif opt.classModel == 'ShakeShake':
    netC = shake_resnet_model.ShakeResNet(opt.nc, opt.RNDepth, opt.RNWidth, nclasses).to(device)
else:
    netC = classifier_model.badGanClass(opt.imageSize, opt.nc, opt.ncf).to(device)
# Generator
netG = generator_model.UNet(opt.imageSize, opt.nc, opt.nc, opt.nz, stn).to(device)
# configure model for multi GPU is multi GPU activated
if torch.cuda.is_available() and opt.cuda and opt.ngpu > 1:
    netG = nn.DataParallel(netG, device_ids=list(range(opt.ngpu)))
    netC = nn.DataParallel(netC, device_ids=list(range(opt.ngpu)))

# load checkpoint if needed
if opt.netC != '':
    netC.load_state_dict(torch.load(opt.netC))
else:
    netC.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
else:
    netG.apply(weights_init)
if not job_running:
    logger(outDir, 'models.txt', str(netC))
    logger(outDir, 'models.txt', str(netG))
print('Models loaded\n')

# define loss
criterion = nn.CrossEntropyLoss().to(device)

# setup optimizer
if opt.optim == 'sgd':
    optimizerC = optim.SGD(netC.parameters(), lr=opt.lrC, momentum=opt.momentum, weight_decay=opt.weight_decay)
elif opt.optim == 'rmsprop':
    optimizerC = optim.RMSprop(netC.parameters(), lr=opt.lrC)
else:
    optimizerC = optim.Adam(netC.parameters(), lr=opt.lrC, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)

# setup LR scheduler
if opt.sched:
    if opt.dataset == 'cifar10':
        if opt.classModel == 'WRN':
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[60,120,160], gamma=0.2)
        elif opt.classModel == 'ShakeShake':
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[60,120,160], gamma=0.2)
        elif opt.classModel == 'ResNet18':
#            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[150,250,350], gamma=0.1)
            scheduler = scheduler.ExponentialLR(optimizerC, gamma=0.5)
        elif opt.classModel == 'ResNetPA':
#            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[150,250,350], gamma=0.1)
            scheduler = scheduler.ExponentialLR(optimizerC, gamma=0.5)
        elif opt.classModel == 'badGAN':
            #            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[100,200,300], gamma=0.1)
            scheduler = scheduler.ExponentialLR(optimizerC, gamma=0.1)
        else:
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[100,200,300], gamma=0.1)
    elif opt.dataset == 'svhn':
        if opt.classModel == 'WRN':
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[60, 120], gamma=0.1)
        if opt.classModel == 'ShakeShake':
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[60, 120], gamma=0.1)
        else:
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[50,100,150], gamma=0.1)
    else:
        scheduler = scheduler.MultiStepLR(optimizerC, milestones=[100,200,300], gamma=0.1)


for epoch in range(currentIteration, opt.niter):
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
            noise = torch.FloatTensor(trainimages.size(0), opt.nz).normal_(0, 0.1).to(device)
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

    # checkpoint each 5 epochs
    if epoch % 5 == 0 and epoch > 0:  # checkpoint every 5 epoch
        jobParams = {}
        jobParams['seed'] = seed
        jobParams['currentIteration'] = epoch
        jobParams['train_idx'] = train_idx.tolist()
        jobParams['val_idx'] = val_idx.tolist()
        with open('{0}/job_running_new'.format(outDir), 'w') as file:
            file.write(json.dumps(jobParams))
        torch.save(netG.state_dict(), '{0}/netG_running_new'.format(outDir))
        torch.save(netC.state_dict(), '{0}/netC_running_new'.format(outDir))
        newJobRunningFile = Path('{0}/job_running_new'.format(outDir))
        newNetG = Path('{0}/netG_running_new'.format(outDir))
        newNetC = Path('{0}/netC_running_new'.format(outDir))
        if newJobRunningFile.exists() and newNetC.exists() and newNetG.exists():
            copyfile('{0}/job_running_new'.format(outDir), '{0}/job_running'.format(outDir))
            copyfile('{0}/netG_running_new'.format(outDir), '{0}/netG_running'.format(outDir))
            copyfile('{0}/netC_running_new'.format(outDir), '{0}/netC_running'.format(outDir))
            os.remove('{0}/job_running_new'.format(outDir))
            os.remove('{0}/netG_running_new'.format(outDir))
            os.remove('{0}/netC_running_new'.format(outDir))
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

# clean temporary file
os.remove('{0}/netG_running'.format(outDir))
os.remove('{0}/netC_running'.format(outDir))
os.remove('{0}/job_running'.format(outDir))
# save final models
torch.save(netC.state_dict(), '{0}/netC_epoch_{1}.pth'.format(outDir, epoch))
