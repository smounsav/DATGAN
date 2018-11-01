import signal,os
import argparse
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
import json
import glob
from shutil import copyfile

from utils import logger as logger
from utils import toOneHot as toOneHot
from cutout import Cutout as Cutout

import models.discriminator as discriminator_model
import models.classifier as classifier_model
import models.wide_resnet as wide_resnet_model
import models.resnet as resnet_model
import models.preact_resnet as preact_resnet_model
import models.shake_resnet as shake_resnet_model
import models.generator as generator_model

parser = argparse.ArgumentParser()
# dataset
parser.add_argument('--dataset', required=True, help='cifar10 | svhn | mnist | stl10 | folder')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--trainsetsize', type=int, help='size of training dataset to use, -1 = full dataset', default=-1)
parser.add_argument('--valratio', type=float, default=0.3, help='ratio of the labeled train dataset to be used as validation set, default = 0.3')
parser.add_argument('--unlbldratio', type=float, default=0, help='ratio of the whole training dataset to be used as unlabeled training set')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network, default = 32')
parser.add_argument('--nc', type=int, default=3, help='number of channels of input image')
# dataset preprocessing
parser.add_argument('--da', action='store_true', help='Whether to apply data augmentation to training dataset')
parser.add_argument('--lightda', action='store_true', help='Whether to apply light data augmentation (only crop) to training dataset')
parser.add_argument('--cutout', action='store_true', help='Whether to apply Cutout on training dataset')
parser.add_argument('--cutoutsize', type=int, default=16, help='Cutout size to apply on training dataset')
# model
parser.add_argument('--classModel', default='badGAN', help='badGAN | WRN | ResNet18 | ResNetPA | ShakeShake')
parser.add_argument('--RNDepth', type=int, default=28, help='Depth factor of the Wide ResNet')
parser.add_argument('--RNWidth', type=int, default=10, help='Width factor of the Wide ResNet')
parser.add_argument('--RNDO', type=float, default=0.3, help='DropOut rate of the Wide ResNet')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ndf', type=int, default=64, help='initial number of filters discriminator')
parser.add_argument('--ngf', type=int, default=64, help='initial number of filters generator')
parser.add_argument('--ncf', type=int, default=64, help='initial number of filters classifier')
parser.add_argument('--diffclass', action='store_true', help='take pairs from different class for DSim')
parser.add_argument('--sameclass', action='store_true', help='take pairs from same class for DSim')
# model ablation
parser.add_argument('--gen', action='store_true', help='Generator will learn a data distribution instead of a transformation')
parser.add_argument('--nostn', action='store_true', help='Deactivate STN in generator')
parser.add_argument('--noDClass', action='store_true', help='Deactivate DClass element')
parser.add_argument('--noClassAdvLoss', action='store_true', help='Deactivate Class adversarial loss for G')
parser.add_argument('--noDDist', action='store_true', help='Deactivate DDist element')
# training
parser.add_argument('--batchSize', type=int, default=64, help='input batch size, default=64')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrDClass', type=float, default=0.0005, help='learning rate for Critic, default=0.0005')
parser.add_argument('--lrDDist', type=float, default=0.0005, help='learning rate for Critic, default=0.0005')
parser.add_argument('--lrG', type=float, default=0.0005, help='learning rate for Generator, default=0.0005')
parser.add_argument('--lrC', type=float, default=0.0005, help='learning rate for Classifier, default=0.0005')
parser.add_argument('--fDClass', type=float, default=0.1, help='learning rate for Critic, default=0.1')
parser.add_argument('--fDDist', type=float, default=0.05, help='learning rate for Critic, default=0.05')
parser.add_argument('--fGCl', type=float, default=0.01, help='learning rate for Critic, default=0.01')
parser.add_argument('--optim', default='adam', help='rmsprop | adam (default is adam)')
parser.add_argument('--optimBDA', action='store_true', help='use same optimizers than in Bayesian DA')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer, default=0.9')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay factor, default=0')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--sched', action='store_true', help='Activate LR rate decay scheduler')
parser.add_argument('--manualSeed', type=int, default=None, help='manuel seed')
# checkpoints
parser.add_argument('--netDClass', default='', help="path to netDClass (to continue training)")
parser.add_argument('--netDDist', default='', help="path to netDDist (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netC', default='', help="path to netC (to continue training)")
# system
parser.add_argument('--workers', type=int, help='number of data loading workers, default = 4', default=4)
parser.add_argument('--outDirPrefix', default=None, help='Additional text for output path')
parser.add_argument('--outDir', default=None, help='Where to store samples and models')
parser.add_argument('--outDirSuffix', default=None, help='Additional text for output path')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pinnedmemory', action='store_true', help='Whether to use GPU pinned memory')

opt = parser.parse_args()

# Assert ratio of unlabeled data and validation data, sum must be lower than 1
assert 0 <= opt.valratio + opt.unlbldratio < 1, 'Error: invalid validation  and unlabeld data ratio. sum should be 0 <= valratio < 1'
assert 0 <= opt.valratio < 1, 'Error: invalid validation data ratio. valratio should be 0 <= valratio < 1'
assert 0 <= opt.unlbldratio < 1, 'Error: invalid unlabeled data ratio. unlbldratio should be 0 <= unlbldratio < 1'

# define output directory
if opt.outDirPrefix is not None:
    outDir = str(opt.outDirPrefix)
else:
    outDir = ''
if opt.outDir is None:
    outDir = outDir + str(opt.dataset).upper() + 'N' + str(opt.trainsetsize) + 'M' + str(opt.classModel) \
             + 'O' + str(opt.optim).upper()
    if opt.optim == 'adam':
        outDir = outDir + 'BT1' + str(opt.beta1).replace('.', '')
    if opt.optim == 'sgd':
        outDir = outDir + 'MOM' + str(opt.momentum).replace('.', '')
    if opt.optim == 'adam' or opt.optim == 'sgd':
        outDir = outDir + 'WD' + str(opt.weight_decay).replace('.', '')
    outDir = outDir + 'B' + str(opt.batchSize) \
             + 'DC' + str(opt.lrDClass).replace('.', '') + 'DD'+ str(opt.lrDDist).replace('.', '') + 'G' + str(opt.lrG).replace('.', '') + 'C' + str(opt.lrC).replace('.', '') \
             + 'LDC' + str(opt.fDClass).replace('.', '') + 'LGC' + str(opt.fGCl).replace('.', '') + 'LDD' + str(opt.fDDist).replace('.', '')
if opt.nostn:
    outDir = outDir + 'NOSTN'
if opt.gen:
    outDir = outDir + 'GEN'
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
newNetDClass = Path('{0}/netDClass_running_new'.format(outDir))
newNetDDist = Path('{0}/netDDist_running_new'.format(outDir))
newNetG = Path('{0}/netG_running_new'.format(outDir))
newNetC = Path('{0}/netC_running_new'.format(outDir))
if newJobRunningFile.exists() and newNetDClass.exists() and newNetDDist.exists() and newNetG.exists() and newNetC.exists():
    copyfile('{0}/job_running_new'.format(outDir), '{0}/job_running'.format(outDir))
    copyfile('{0}/netDClass_running_new'.format(outDir), '{0}/netDClass_running'.format(outDir))
    copyfile('{0}/netDDist_running_new'.format(outDir), '{0}/netDDist_running'.format(outDir))
    copyfile('{0}/netG_running_new'.format(outDir), '{0}/netG_running'.format(outDir))
    copyfile('{0}/netC_running_new'.format(outDir), '{0}/netC_running'.format(outDir))
    os.remove('{0}/job_running_new'.format(outDir))
    os.remove('{0}/netDClass_running_new'.format(outDir))
    os.remove('{0}/netDDist_running_new'.format(outDir))
    os.remove('{0}/netG_running_new'.format(outDir))
    os.remove('{0}/netC_running_new'.format(outDir))

# if job is restarted after interruption, reload parameters
ongoingFile = Path('{0}/job_running'.format(outDir))
if ongoingFile.exists():
    job_running =True
    with open('{0}/job_running'.format(outDir)) as file:
        jobParams = json.loads(file.read())
    opt.manuelSeed = jobParams['Seed']
    currentIteration = jobParams['currentIteration'] + 1
    train_idx = torch.tensor(jobParams['train_idx'])
    train_idx_2 = torch.tensor(jobParams['train_idx_2'])
    val_idx = torch.tensor(jobParams['val_idx'])
    unlbl_idx = torch.tensor(jobParams['unlbl_idx'])
    opt.netDClass = outDir + '/netDClass_running'
    opt.netDDist = outDir + '/netDDist_running'
    opt.netG = outDir + '/netG_running'
    opt.netC = outDir + '/netC_running'
    print('Restarting job\n')
else:
    # clean directory if already existing
    filelist = glob.glob('{0}/*'.format(outDir))
    for file in filelist:
        os.remove(file)
    # log command line parameters
    print(opt)
    logger(outDir, 'parameters.txt', str(opt))
    print('Starting job\n')

# define and log random seed for reproductibility
if opt.manualSeed is not None:
    manualSeed = opt.manualSeed
else:
    manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if not job_running:
    print("Random Seed: ", manualSeed)
    logger(outDir, 'parameters.txt', 'Random seed: ' + str(manualSeed)) # log random seed

# start cudnn autotuner
cudnn.benchmark = True
# make training deterministic/reproductible
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
if opt.dataset == 'mnist':
    transformTrain.transforms.append(transforms.Pad(2))
if opt.lightda:
    if opt.dataset in ['mnist']:
        transformTrain.transforms.append(transforms.RandomCrop(32, padding=4))
    elif opt.dataset in ['cifar10']:
        transformTrain.transforms.append(transforms.RandomCrop(32, padding=4))
if opt.da:
    if opt.dataset in ['mnist']:
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

# Separate training data into fully labeled training, fully labeled validation and unlabeled dataset
len_train = len(trainset)
if opt.dataset in ['svhn', 'stl10']:
    class_dict = {i: trainset.labels[i] for i in range(len(trainset.labels))}  # to speed up get item in Ddist
else:
    class_dict = {i:trainset.train_labels[i] for i in range(len(trainset.train_labels))} # to speed up get item in Ddist
indices_train = list(range(len_train))
if not job_running:
    shuffled_indices_train = torch.randperm(len_train)
#    shuffled_indices_train = indices_train[torch.randperm(len(indices_train))]
#    random.shuffle(shuffled_indices_train)
    if opt.trainsetsize != -1:
        assert 0 < opt.trainsetsize <= len_train, 'Error: invalid training dataset size.'
        indices_train_reduced = shuffled_indices_train[:opt.trainsetsize]
    else:
        indices_train_reduced = shuffled_indices_train
    len_train_reduced = len(indices_train_reduced)
    # Training dataset is reduced to [Unlabeled samples:Training labeled samples:Validation labeled samples]
    split_lbl_unlbl = int(opt.unlbldratio * len_train_reduced)
    split_train_val = int(opt.valratio * len_train_reduced)
    unlbl_idx = indices_train_reduced[:split_lbl_unlbl]
    train_idx, val_idx = indices_train_reduced[split_lbl_unlbl + split_train_val:], indices_train_reduced[split_lbl_unlbl:split_lbl_unlbl + split_train_val]
    train_idx_2 = train_idx[torch.randperm(len(train_idx))]
#    print(len(train_idx))
#    print(len(val_idx))
#    print(len(unlbl_idx))
    logger(outDir, 'parameters.txt', 'Train dataset: ' + str(train_idx.tolist())) # Save samples used for training
    logger(outDir, 'parameters.txt', 'Validation dataset: ' + str(val_idx.tolist())) # Save samples used for validation
    logger(outDir, 'parameters.txt', 'Unlabeled dataset: ' + str(unlbl_idx.tolist())) # Save samples used as unlabeled samples

# define data samplers
train_sampler = SubsetRandomSampler(train_idx)
train_sampler_2 = SubsetRandomSampler(train_idx_2)
val_sampler = SubsetRandomSampler(val_idx)
unlbl_sampler = SubsetRandomSampler(unlbl_idx)


def worker_init(x):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# define data loaders
assert trainset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, sampler=train_sampler,
                                          shuffle=False, num_workers=opt.workers, pin_memory=pinned_memory, worker_init_fn=worker_init)
trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, sampler=train_sampler_2,
                                          shuffle=False, num_workers=opt.workers, pin_memory=pinned_memory, worker_init_fn=worker_init)

valloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, sampler=val_sampler,
                                          shuffle=False, num_workers=opt.workers, pin_memory=pinned_memory, worker_init_fn=worker_init)

batchSizeUnlbl = int(opt.batchSize * (len(unlbl_idx) / len(train_idx)))
if batchSizeUnlbl == 0:
    batchSizeUnlbl = opt.batchSize
unlblloader = torch.utils.data.DataLoader(trainset, batch_size=batchSizeUnlbl, sampler=unlbl_sampler,
                                          shuffle=False, num_workers=opt.workers, pin_memory=pinned_memory, worker_init_fn=worker_init)

assert testset
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=opt.workers, worker_init_fn=worker_init)

nz = int(opt.nz)
nc = int(opt.nc)
ndf = int(opt.ndf)
ngf = int(opt.ngf)
ncf = int(opt.ncf)

# weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
#        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
#        m.weight.data.normal_(0, 0.05)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.05)
        m.bias.data.fill_(0)

# Model initialisation
# Discriminator
netDClass = discriminator_model.badGanDClass(opt.imageSize, nc, ndf, nclasses).to(device)
netDDist = discriminator_model.badGanDDist(opt.imageSize, nc, ndf).to(device)
# Generator
if opt.gen:
    netG = generator_model.badGanGen(opt.imageSize, nc, nz, nclasses).to(device)
else:
    netG = generator_model.UNet(opt.imageSize, nc, nc, nz, not opt.nostn).to(device)
# Classifier
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

if torch.cuda.is_available() and opt.cuda and opt.ngpu > 1:
    netDClass = nn.DataParallel(netDClass, device_ids=list(range(opt.ngpu)))
    netDDist = nn.DataParallel(netDDist, device_ids=list(range(opt.ngpu)))
    netG = nn.DataParallel(netG, device_ids=list(range(opt.ngpu)))
    netC = nn.DataParallel(netC, device_ids=list(range(opt.ngpu)))

# Load model checkpoint if needed
if opt.netDClass != '':
    netDClass.load_state_dict(torch.load(opt.netDClass))
else:
    netDClass.apply(weights_init)
if opt.netDDist != '':
    netDDist.load_state_dict(torch.load(opt.netDDist))
else:
    netDDist.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
else:
    netG.apply(weights_init)
if opt.netC != '':
    netC.load_state_dict(torch.load(opt.netC))
else:
    netC.apply(weights_init)
if not job_running:
    logger(outDir, 'models.txt', str(netDClass))
    logger(outDir, 'models.txt', str(netC))
    logger(outDir, 'models.txt', str(netG))
    logger(outDir, 'models.txt', str(netDDist))
print('Models loaded\n')

# define loss
CELoss = nn.CrossEntropyLoss().to(device)
BCEWithLogitsLoss = nn.BCEWithLogitsLoss().to(device)

# setup optimizer
if opt.optimBDA:
    optimizerDClass = optim.SGD(netDClass.parameters(), lr=0.01)
    optimizerDDist = optim.Adam(netDDist.parameters(), lr=opt.lrDDist, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.9, 0.999))
    optimizerC = optim.Adadelta(netC.parameters(), lr=1.0, rho=0.95, eps=1e-08, weight_decay=0)
if opt.optim == 'rmsprop':
    optimizerDClass = optim.RMSprop(netDClass.parameters(), lr=opt.lrDClass)
    optimizerDDist = optim.RMSprop(netDDist.parameters(), lr=opt.lrDDist)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)
    optimizerC = optim.RMSprop(netC.parameters(), lr=opt.lrC)
elif opt.optim == 'sgd':
    optimizerDClass = optim.Adam(netDClass.parameters(), lr=opt.lrDClass, betas=(opt.beta1, 0.999))
    optimizerDDist = optim.Adam(netDDist.parameters(), lr=opt.lrDDist, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    optimizerC = optim.SGD(netC.parameters(), lr=opt.lrC, momentum=opt.momentum, weight_decay=opt.weight_decay)
else:
    optimizerDClass = optim.Adam(netDClass.parameters(), lr=opt.lrDClass, betas=(opt.beta1, 0.999))
    optimizerDDist = optim.Adam(netDDist.parameters(), lr=opt.lrDDist, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    optimizerC = optim.Adam(netC.parameters(), lr=opt.lrC, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)


# setup LR scheduler
if opt.sched:
    if opt.dataset == 'cifar10':
        if opt.classModel == 'WRN':
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[250, 375], gamma=0.1)
        elif opt.classModel == 'ShakeShake':
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[60,120,160], gamma=0.2)
        elif opt.classModel == 'ResNet18':
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[150,250,350], gamma=0.1)
        elif opt.classModel == 'ResNetPA':
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[150,250,350], gamma=0.1)
        elif opt.classModel == 'badGAN':
#            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[100,200,300], gamma=0.1)
            scheduler = scheduler.ExponentialLR(optimizerC, gamma=0.999)
        else:
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[100,200,300], gamma=0.1)
    elif opt.dataset == 'svhn':
        if opt.classModel == 'WRN':
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[60, 120], gamma=0.1)
        elif opt.classModel == 'ShakeShake':
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[60, 120], gamma=0.1)
        else:
            scheduler = scheduler.MultiStepLR(optimizerC, milestones=[50,100,150], gamma=0.1)
    else:
        scheduler = scheduler.MultiStepLR(optimizerC, milestones=[100,200,300], gamma=0.1)

for epoch in range(currentIteration, opt.niter):
    if opt.sched:
        scheduler.step()
    netC.train()
    # Initialize losses
    totalLossD = 0.0
    totalLossDClass, totalLossDClassReal, totalLossDClassGen, totalLossDClassUnlbl = 0.0, 0.0, 0.0, 0.0
    totalLossDDist, totalLossDDistReal, totalLossDDistGen = 0.0, 0.0, 0.0
    totalLossG, totalLossGClass, totalLossGDist, totalLossGClEnt, totalLossGClCE, totalLossGCl2B = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    totalLossC, totalLossCCEReal, totalLossCCEGen, totalLossCClassUnlbl = 0.0, 0.0, 0.0, 0.0

    if opt.unlbldratio > 0:
        data = zip(trainloader, trainloader2, unlblloader)
    else:
        data = zip(trainloader, trainloader2)
    for i, item in enumerate(data):
        if len(unlblloader) > 0:
            lbldata, lbldata2, unlbldata = item
        else:
            lbldata, lbldata2 = item

        ############################
        # (0) Sample training sets
        ###########################

        # Sample batch of real labeled training images to transform
        trainrealimages, trainreallabels = lbldata
        batch_size_lbl = trainrealimages.size(0)
        trainrealimages =  trainrealimages.to(device)
        trainreallabels = trainreallabels.to(device)
        # converting labels to one hot encoding form
        onehottrainreallabels = toOneHot(trainreallabels, nclasses)

        # Sample batch of real labeled training images to train DClass
        trainrealimages2, trainreallabels2 = lbldata2
        batch_size_lbl2 = trainrealimages2.size(0)
        trainrealimages2 =  trainrealimages2.to(device)
        trainreallabels2 = trainreallabels2.to(device)
        # converting labels to one hot encoding form
        onehottrainreallabels2 = toOneHot(trainreallabels2, nclasses)

        # Generate noise to transform images
        noise = torch.FloatTensor(batch_size_lbl, nz).normal_(0, 0.1).to(device)
        label_1 = torch.FloatTensor(batch_size_lbl).fill_(1).to(device)
        label_0 = torch.FloatTensor(batch_size_lbl).fill_(0).to(device)

        # Sample batch of real unlabeled training images
        if opt.unlbldratio > 0:
            trainrealunlblimages, _ = unlbldata
            trainrealunlblimages = trainrealunlblimages.to(device)
            label_1u = torch.FloatTensor(trainrealunlblimages.size(0)).fill_(1).to(device)
            label_0u = torch.FloatTensor(trainrealunlblimages.size(0)).fill_(0).to(device)

        ##############################
        # (1.1) Update DClass network
        ##############################
        for p in netG.parameters():  # to avoid computation
            p.requires_grad = False
        for p in netC.parameters():  # to avoid computation
            p.requires_grad = False
        for p in netDClass.parameters():
            p.requires_grad = True
        for p in netDDist.parameters(): # to avoid computation
            p.requires_grad = False

        netDClass.zero_grad()

        # train with real
        output_1 = netDClass(trainrealimages2, onehottrainreallabels2)
        lossDClass_real = BCEWithLogitsLoss(output_1, label_1)
        totalLossDClassReal += lossDClass_real.item()

        # train with generated
        if opt.gen:
            traingenimages = netG(onehottrainreallabels, noise)
        else:
            traingenimages = netG(trainrealimages, noise)

        output_0 = netDClass(traingenimages, onehottrainreallabels)
        lossDClass_gen = BCEWithLogitsLoss(output_0, label_0)
        totalLossDClassGen += lossDClass_gen.item()

        # train with unlabeled
        if opt.unlbldratio > 0:
            predlabelsunlbl = netC(trainrealunlblimages)
            predclassunlbltrain = predlabelsunlbl.max(1, keepdim=True)[1]
            onehottrainrealunlbllabels = toOneHot(predclassunlbltrain.squeeze(1), nclasses)

            output_u = netDClass(trainrealunlblimages, onehottrainrealunlbllabels)
            lossDClass_unlbl = BCEWithLogitsLoss(output_u, label_0u)
            totalLossDClassUnlbl += lossDClass_unlbl.item()
            lossDClass = (lossDClass_real + lossDClass_gen + lossDClass_unlbl) / 1
        else:
            lossDClass = (lossDClass_real + lossDClass_gen) / 1
        lossDClass.backward()
        totalLossDClass = totalLossDClassReal + totalLossDClassGen + totalLossDClassUnlbl
        optimizerDClass.step()

        #############################
        # (1.2) Update DSim network
        #############################
        for p in netDClass.parameters(): # to avoid computation
            p.requires_grad = False
        for p in netDDist.parameters():
            p.requires_grad = True

        netDDist.zero_grad()

        # Maximize distance between input sample and generated sample
        output_dist_0 = netDDist(trainrealimages, traingenimages)
        lossDDist_real = BCEWithLogitsLoss(output_dist_0, label_0)
        totalLossDDistReal += lossDDist_real.item()

        # Minimize distance between 2 true sample from same class
        reftrainimages = trainrealimages.clone()
        for idx in range(reftrainimages.size(0)):
            if opt.diffclass:
                reftrainimages[idx], _ = trainset.__getitem__(train_idx[index])
            else:
                found = False
                while not found:
                    index = random.randint(0, len(train_idx) - 1)
                    if class_dict[train_idx[index].item()] == trainreallabels[idx].item():
                        found = True
                        reftrainimages[idx], _ = trainset.__getitem__(train_idx[index])

        output_dist_1 = netDDist(trainrealimages, reftrainimages)
        lossDDist_gen = BCEWithLogitsLoss(output_dist_1, label_1)
        totalLossDDistGen += lossDDist_gen.item()
        # Loss
        lossDDist = (lossDDist_real + lossDDist_gen) / 1
        lossDDist.backward()
        optimizerDDist.step()
        totalLossDDist = (totalLossDDistReal + totalLossDDistGen) / 1

        totalLossD = totalLossDClass + totalLossDDist

        ############################
        # (2) Update G network
        ###########################
        for p in netDDist.parameters(): # to avoid computation
            p.requires_grad = False
        for p in netG.parameters():
            p.requires_grad = True
        netG.zero_grad()
        # Class loss
        if opt.gen:
            traingenimages = netG(onehottrainreallabels, noise.squeeze(-1).squeeze(-1))
        else:
            traingenimages = netG(trainrealimages, noise)
        output_0 = netDClass(traingenimages, onehottrainreallabels)
        lossGClass =  BCEWithLogitsLoss(output_0, label_1)
        totalLossGClass += lossGClass.item()
        # Distance loss
        output_1 = netDDist(trainrealimages, traingenimages)
        lossGDist = BCEWithLogitsLoss(output_1, label_1)
        totalLossGDist += lossGDist.item()

        logitsgenlabels = netC(traingenimages)
        # Label entropy loss term
#        lossGClEnt = (nn.functional.softmax(logitsgenlabels, dim=1).mul(nn.functional.log_softmax(logitsgenlabels, dim=1))).sum(dim=1).mean()
#        totalLossGClEnt += lossGEnt.item()

        # Label cross entropy  loss term
        lossGClCE = CELoss((1 - nn.functional.softmax(logitsgenlabels, dim=1)), trainreallabels)
        totalLossGClCE += lossGClCE.item()

        # Label loss term - Marcos proposition
#        probgenlabels = nn.functional.softmax(logitsgenlabels, dim=1)
#        pgt = probgenlabels.mul(onehottrainreallabels).sum(dim=1)
#        p2best, _ = torch.topk(probgenlabels,2, dim=1)
#        pbest = torch.empty(p2best.size(0)).to(device)
#        for idx in range(len(pbest)):
#            if p2best[idx][0] == pgt[idx]:
#                pbest[idx] = p2best[idx][1]
#            else:
#                pbest[idx] = pgt[idx]
#        lossGCl2B = - torch.log( (pgt - pbest).abs() ).mean()
#        totalLossGCl2B += lossGCl2B.item()

        lossGCl = lossGClCE # + lossGClEnt + lossGCl2B
        totalLossGCl = totalLossGClCE  # + totalLossGClEnt + totalLossGClCE totalLossGCl2B


        # Loss
        lossG =  opt.fDClass * lossGClass + opt.fGCl * lossGCl + opt.fDDist * lossGDist
        totalLossG = opt.fDClass * totalLossGClass + opt.fGCl * totalLossGCl + opt.fDDist * totalLossGDist
        lossG.backward()
        optimizerG.step()

        ############################
        # (3) Update C network
        ###########################
        for p in netG.parameters():
            p.requires_grad = False # to avoid computation

        for p in netC.parameters():
            p.requires_grad = True

        netC.train()
        netC.zero_grad()

        # train with real
        predtrainreallabels = netC(trainrealimages2)
        lossC_CE_real = CELoss(predtrainreallabels, trainreallabels2)
        totalLossCCEReal += lossC_CE_real.item()

        # train with fake
#        traingenimages = netG(trainrealimages, noise)
        predtraingenlabels = netC(traingenimages.detach())
        lossC_CE_gen = CELoss(predtraingenlabels, trainreallabels)
        totalLossCCEGen += lossC_CE_gen.item()
        lossC_CE = lossC_CE_real + lossC_CE_gen

        # train with unlabeled
        if opt.unlbldratio > 0:
            predtrainrealunlbllabels = netC(trainrealunlblimages)
            predlabelsunlbl_softmaxed = nn.functional.softmax(predtrainrealunlbllabels, dim=1)
            predscoreunlbltrain = predlabelsunlbl_softmaxed.max(1, keepdim=True)[0]
            predclassunlbltrain = predlabelsunlbl_softmaxed.max(1, keepdim=True)[1]
            onehottrainrealunlbllabels = toOneHot(predclassunlbltrain.squeeze(1), nclasses)
            output_u = (netDClass(trainrealunlblimages, onehottrainrealunlbllabels)).unsqueeze( -1)
            lossCClass_unlbl = predscoreunlbltrain.mul(torch.nn.functional.logsigmoid(-output_u)).mean()
            totalLossCClassUnlbl += lossCClass_unlbl.item()
            LossC = (lossC_CE + lossCClass_unlbl) / 1
        else:
            LossC = lossC_CE / 1
        LossC.backward()
        totalLossC = (totalLossCCEReal + totalLossCCEGen + totalLossCClassUnlbl) / 1
        optimizerC.step()

        if epoch % 10 == 0 and epoch > 0:  # print and save loss every 10 epochs
            nbcorrecttrain = 0.0
            totaltrain = 0.0
            trainacc = 0.0
            with torch.no_grad():
                # get the index of the max log-probability to get the label
                predclassrealtrain = predtrainreallabels.max(1, keepdim=True)[1]
                # count the number of samples correctly classified
                nbcorrecttrain += predclassrealtrain.eq(trainreallabels2.view_as(predclassrealtrain)).sum()
                totaltrain += trainreallabels2.size(0)
                trainacc = 100 * nbcorrecttrain.item() / totaltrain

    if epoch % 10 == 0 and epoch >= 0: # print and save loss every 10 epochs

        # Test C on validation set
        netC.eval()
        with torch.no_grad():
            nbcorrectval = 0.0
            totalval = 0.0
            valacc = 0.0
            if opt.valratio > 0:
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
                    nbcorrectval += valpred.eq(vallabels.view_as(valpred)).sum()
                    totalval += vallabels.size(0)
                    valacc = 100 * nbcorrectval.item() / totalval

            nbcorrecttest = 0.0
            totaltest = 0.0
            testacc = 0.0
            for testdata in testloader:
                # Get test images
                testimages, testlabels = testdata
                # activate cuda version of the variables if cuda is activated
                testimages, testlabels = testimages.to(device), testlabels.to(device)
                # Calculate scores
                testoutput = netC(testimages)
                # get the index of the max log-probability to get the label
                testpred = testoutput.max(1, keepdim=True)[1]
                # count the number of samples correctly classified
                nbcorrecttest += testpred.eq(testlabels.view_as(testpred)).sum()
                totaltest += testlabels.size(0)
                testacc = 100 * nbcorrecttest.item() / totaltest

    # checkpoint each 5 epochs
    if epoch % 5 == 0 and epoch > 0:  # print every 10 epoch
        jobParams = {}
        jobParams['Seed'] = manualSeed
        jobParams['currentIteration'] = epoch
        jobParams['train_idx'] = train_idx.tolist()
        jobParams['train_idx_2'] = train_idx_2.tolist()
        jobParams['val_idx'] = val_idx.tolist()
        jobParams['unlbl_idx'] = unlbl_idx.tolist()
        with open('{0}/job_running_new'.format(outDir), 'w') as file:
            file.write(json.dumps(jobParams))
        torch.save(netDClass.state_dict(), '{0}/netDClass_running_new'.format(outDir))
        torch.save(netDDist.state_dict(), '{0}/netDDist_running_new'.format(outDir))
        torch.save(netG.state_dict(), '{0}/netG_running_new'.format(outDir))
        torch.save(netC.state_dict(), '{0}/netC_running_new'.format(outDir))
        newJobRunningFile = Path('{0}/job_running_new'.format(outDir))
        newNetDClass = Path('{0}/netDClass_running_new'.format(outDir))
        newNetDDist = Path('{0}/netDDist_running_new'.format(outDir))
        newNetG = Path('{0}/netG_running_new'.format(outDir))
        newNetC = Path('{0}/netC_running_new'.format(outDir))
        if newJobRunningFile.exists() and newNetDClass.exists() and newNetDDist.exists() and newNetG.exists() and newNetC.exists():
            copyfile('{0}/job_running_new'.format(outDir), '{0}/job_running'.format(outDir))
            copyfile('{0}/netDClass_running_new'.format(outDir), '{0}/netDClass_running'.format(outDir))
            copyfile('{0}/netDDist_running_new'.format(outDir), '{0}/netDDist_running'.format(outDir))
            copyfile('{0}/netG_running_new'.format(outDir), '{0}/netG_running'.format(outDir))
            copyfile('{0}/netC_running_new'.format(outDir), '{0}/netC_running'.format(outDir))
            os.remove('{0}/job_running_new'.format(outDir))
            os.remove('{0}/netDClass_running_new'.format(outDir))
            os.remove('{0}/netDDist_running_new'.format(outDir))
            os.remove('{0}/netG_running_new'.format(outDir))
            os.remove('{0}/netC_running_new'.format(outDir))
    if epoch % 10 == 0 and epoch > 0:  # print on screen and save loss every 10 epochs
    # Print loss on screen for monitoring
        loss = '[{0}/{1}] lossD {2} lossDClass: {3} lossDDist {4} lossG: {5} lossGClass: {6} lossGDist: {7} lossGClent: {8} lossGClCE: {9} lossGCl2B: {10} lossC: {11} trainacc {12} valacc {13} testacc {14}'.format(
            epoch, opt.niter,
            totalLossD, totalLossDClass, totalLossDDist,
            totalLossG, totalLossGClass, totalLossGDist, totalLossGClEnt, totalLossGClCE, totalLossGCl2B,
            totalLossC,
            trainacc, valacc, testacc)
        print(loss)

        # save loss in a log file
        logger(outDir, 'loss.txt', str(loss))

    if epoch % 50 == 0 and epoch > 0: # save some examples of original and transformed images
        traingenimages = traingenimages.mul(0.5).add(0.5)
        vutils.save_image(traingenimages, '{0}/{1}_generated_samples.png'.format(outDir, epoch))
        realimages = trainrealimages.mul(0.5).add(0.5)
        vutils.save_image(realimages, '{0}/{1}_real_samples.png'.format(outDir, epoch))

# clean temporary file
os.remove('{0}/netDClass_running'.format(outDir))
os.remove('{0}/netDDist_running'.format(outDir))
os.remove('{0}/netG_running'.format(outDir))
os.remove('{0}/netC_running'.format(outDir))
os.remove('{0}/job_running'.format(outDir))
# save final models
torch.save(netDClass.state_dict(), '{0}/netDClass_epoch_{1}.pth'.format(outDir, epoch))
torch.save(netDDist.state_dict(), '{0}/netDDist_epoch_{1}.pth'.format(outDir, epoch))
torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(outDir, epoch))
torch.save(netC.state_dict(), '{0}/netC_epoch_{1}.pth'.format(outDir, epoch))
