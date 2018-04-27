from __future__ import print_function
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import datetime

import models.discriminator as discriminator_model
import models.classifier as classifier_model
import models.generator as generator_model

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--ngf', type=int, default=64, help='final number of filters generator')
parser.add_argument('--ndf', type=int, default=64, help='initial number of filters discriminator')
parser.add_argument('--ncf', type=int, default=64, help='initial number of filters classifier')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--lrC', type=float, default=0.00005, help='learning rate for Classifier, default=0.00005')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.5)')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD_Class', default='', help="path to netD_Class (to continue training)")
parser.add_argument('--netD_Dist', default='', help="path to netD_Dist (to continue training)")
parser.add_argument('--netC', default='', help="path to netC (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--outDir', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--kldiv', action='store_true', help='Whether to use KL Divergence (default is WGAN)')
opt = parser.parse_args()
print(opt)

if opt.outDir is None:
    now = datetime.datetime.now().timetuple()
    opt.outDir = 'run-' + str(now.tm_year) + str(now.tm_mon) + str(now.tm_mday) + str(now.tm_hour) + str(now.tm_min) + str(now.tm_sec)
os.system('mkdir {0}'.format(opt.outDir))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# define preprocessing transformations
transform=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize images between -1 and 1

# define datset and initialise dataloader
trainset = dset.ImageFolder(root=opt.dataroot + '/train', transform=transform)
testset = dset.ImageFolder(root=opt.dataroot + '/test', transform=transform)

assert trainset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=opt.workers)

print(len(trainloader))
nclasses = len(trainset.classes)

assert testset
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=opt.workers)
print(len(testloader))

ngf = int(opt.ngf)
ndf = int(opt.ndf)
ncf = int(opt.ncf)
nc = int(opt.nc)
# custom weights initialization called on netG and netD_Class
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

# Model initialisation
# Discriminator
netD_Class = discriminator_model.D_Class(opt.imageSize, nc, ndf, nclasses)
netD_Class.apply(weights_init)
netD_Dist = discriminator_model.D_Dist(opt.imageSize, nc, ndf)
netD_Dist.apply(weights_init)
# Generator
netG = generator_model.UNet(nc,nc)
netG.apply(weights_init)
# Classifier
netC = classifier_model.cnnClass(nc, ncf)
netC.apply(weights_init)

# Load model checkpoint if needed
if opt.netD_Class != '':
    netD_Class.load_state_dict(torch.load(opt.netD_Class))
print(netD_Class)
if opt.netD_Dist != '':
    netD_Dist.load_state_dict(torch.load(opt.netD_Dist))
print(netD_Dist)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)
if opt.netC != '':
    netD_Class.load_state_dict(torch.load(opt.netC))
print(netC)

# define helpers for optimisation
one = torch.FloatTensor([1])
mone = one * -1

# setup optimizer
if opt.adam:
    optimizerD_Class = optim.Adam(netD_Class.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerD_Dist = optim.Adam(netD_Dist.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    optimizerC = optim.Adam(netC.parameters(), lr=opt.lrC, betas=(opt.beta1, 0.999))
else:
    optimizerD_Class = optim.RMSprop(netD_Class.parameters(), lr=opt.lrD)
    optimizerD_Dist = optim.RMSprop(netD_Dist.parameters(), lr=opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)
    optimizerC = optim.RMSprop(netC.parameters(), lr=opt.lrC)

# define classification loss
CELoss = nn.CrossEntropyLoss()
if opt.kldiv:
    BCELoss = nn.BCEWithLogitsLoss()

# activate cuda acceleration if available
if opt.cuda:
    netD_Class.cuda()
    netD_Dist.cuda()
    netG.cuda()
    netC.cuda()
    one, mone = one.cuda(), mone.cuda()
    CELoss = CELoss.cuda()
    if opt.kldiv:
        BCELoss = BCELoss.cuda()
#    imageDist = imageDist.cuda()

gen_iterations = 0
for epoch in range(opt.niter):
    data_iter = iter(trainloader)
    i = 0

    # Initialize losses
    lossD = 0.0
    lossD_Class = 0.0
    total_lossD_Class_real = 0.0
    total_lossD_Class_gen = 0.0
    lossD_Dist = 0.0
    total_lossD_Dist_real = 0.0
    total_lossD_Dist_gen = 0.0
    lossG = 0.0
    total_lossG_Class = 0.0
    total_lossG_Dist = 0.0
    total_lossG_CE = 0.0
    lossC = 0.0
    total_lossC_CE_real = 0.0
    total_lossC_CE_gen = 0.0

    while i < len(trainloader):

        ############################
        # (1) Update D network
        ###########################
        for p in netD_Class.parameters():
            p.requires_grad = True
        for p in netD_Dist.parameters():
            p.requires_grad = True
        for p in netG.parameters(): # to avoid computation
            p.requires_grad = False
        for p in netC.parameters(): # to avoid computation
            p.requires_grad = False

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = opt.Diters
        j = 0
        while j < Diters and i < len(trainloader):
            j += 1
            i += 1

            # clamp parameters to a cube
            if not opt.kldiv:
                for p in netD_Class.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            try:
                data = data_iter.next()
            except StopIteration:
                data_iter = iter(trainloader)
                data = next(data_iter)

            # Sample batch of real training images
            trainrealimages, trainreallabels = data
            batch_size = trainrealimages.size(0)
            trainrealimages = Variable(trainrealimages)
            trainreallabels = Variable(trainreallabels)
            if opt.kldiv:
                label_1 = Variable(torch.FloatTensor(batch_size).fill_(1))
            # converting labels to one hot encoding form
            onehotlabelssupport = torch.FloatTensor(batch_size, nclasses).zero_()
            onehottrainreallabels = Variable(onehotlabelssupport.scatter_(1, trainreallabels.unsqueeze(1).data, 1))

            if opt.cuda:
                trainrealimages = trainrealimages.cuda()
                trainreallabels, onehottrainreallabels = trainreallabels.cuda(), onehottrainreallabels.cuda()
                if opt.kldiv:
                    label_1 = label_1.cuda()

            ##############################
            # (1.1) Update D_Class network
            ##############################
            netD_Class.zero_grad()

            output_1 = netD_Class(trainrealimages, onehottrainreallabels)
            # train with real
            if opt.kldiv:
                lossD_Class_real = BCELoss(output_1, label_1)
            else:
                lossD_Class_real = output_1.mean()
            lossD_Class_real.backward(mone)

            # train with generated
            traingenimages = netG(trainrealimages)
            output_0 = netD_Class(traingenimages, onehottrainreallabels)
            if opt.kldiv:
                lossD_Class_gen = BCELoss(output_0, label_1)
            else:
                lossD_Class_gen = output_0.mean()
            lossD_Class_gen.backward(one)

            # train with unlabeled
            # complete with unlabeled label

            #lossD_unlabeled = netD_Class(traingenimages, trainreallabels)
            #lossD_unlabeled.backward(mone)
            total_lossD_Class_real += - lossD_Class_real.data[0]
            total_lossD_Class_gen += lossD_Class_gen.data[0]
            lossD_Class += total_lossD_Class_real + total_lossD_Class_gen #- lossD_unlabeled
            optimizerD_Class.step()

            #############################
            # (1.2) Update D_Dist network
            #############################
            netD_Dist.zero_grad()

            # Minimize distance between input sample and a sample from same class
            reftrainimages = trainrealimages.clone()
            for i in range(reftrainimages.size(0)):
                found = False
                while not found:
                    index = random.randint(0, len(trainset)-1)
                    selected_image, selected_label = trainset.__getitem__(index)
                    selected_image = Variable(selected_image)
                    if opt.cuda:
                        selected_image = selected_image.cuda()
                    if selected_label != trainreallabels.data[i]:
                        found = True
                        reftrainimages[i] = selected_image

            outpout = netD_Dist(trainrealimages, reftrainimages)
            if opt.kldiv:
                lossD_Dist_real = BCELoss(outpout, label_1)
            else:
                lossD_Dist_real = outpout.mean()
            lossD_Dist_real.backward(mone)

            # Maximize distance between input sample and generated sample
            outpout = netD_Dist(trainrealimages, traingenimages)
            if opt.kldiv:
                lossD_Dist_gen = BCELoss(outpout, label_1)
            else:
                lossD_Dist_gen = outpout.mean()
            lossD_Dist_gen.backward(one)

            # Loss
            total_lossD_Dist_real += - lossD_Dist_real.data[0]
            total_lossD_Dist_gen += lossD_Dist_gen.data[0]
            lossD_Dist += total_lossD_Dist_real + total_lossD_Dist_gen
            optimizerD_Dist.step()

            lossD += lossD_Class + lossD_Dist

        ############################
        # (2) Update G network
        ###########################
        for p in netD_Dist.parameters():
            p.requires_grad = False # to avoid computation
        for p in netD_Class.parameters():
            p.requires_grad = False # to avoid computation
        for p in netG.parameters():
            p.requires_grad = True
        for p in netC.parameters():
            p.requires_grad = False # to avoid computation

        netG.zero_grad()
        traingenimages = netG(trainrealimages)
        output_0 = netD_Class(traingenimages, onehottrainreallabels)
        # True/Fake Loss
        if opt.kldiv:
           lossG_Class = BCELoss(output_0, label_1)
        else:
           lossG_Class = output_0.mean()
        lossG_Class.backward(mone, retain_graph=True)

#        lossG_Dist = imageDist(traingenimages.view(traingenimages.size(0), -1), reftrainimages.view(traingenimages.size(0), -1)).mean()
        output_1 = netD_Dist(trainrealimages, traingenimages)
        # True/Fake Loss
        if opt.kldiv:
            lossG_Dist = BCELoss(output_1, label_1)
        else:
            lossG_Dist= output_1.mean()
        lossG_Dist.backward(mone, retain_graph=True)
        # Label cross entropy  loss term
        logitsgenlabels = netC(traingenimages)
        maskedlogitsgenlabels = nn.functional.softmax(logitsgenlabels, dim=1).mul(onehottrainreallabels).sum(dim=1)
        if opt.kldiv:
            lossG_CE = BCELoss(maskedlogitsgenlabels, label_1)
        else:
            lossG_CE = maskedlogitsgenlabels.mean()
        lossG_CE.backward(one)

        # Loss
        total_lossG_Class += - lossG_Class.data[0]
        total_lossG_Dist += - lossG_Dist.data[0]
        total_lossG_CE += lossG_CE.data[0]
        lossG += total_lossG_Class + total_lossG_Dist + total_lossG_CE

        optimizerG.step()

        ############################
        # (3) Update C network
        ###########################
        for p in netD_Dist.parameters():
            p.requires_grad = False # to avoid computation
        for p in netD_Class.parameters():
            p.requires_grad = False # to avoid computation
        for p in netG.parameters():
            p.requires_grad = False # to avoid computation
        for p in netC.parameters():
            p.requires_grad = True

        netC.train()
        netC.zero_grad()

        # train with real
        predtrainreallabels = netC(trainrealimages)
        nbcorrectlabel = 0
        totalnblabels = 0
        if epoch % 10 == 0 and epoch > 0:  # print and save loss every 10 epochs
            # get the index of the max log-probability to get the label
            predclassrealtrain = predtrainreallabels.data.max(1, keepdim=True)[1]
            # count the number of samples correctly classified
            nbcorrectlabel += predclassrealtrain.eq(trainreallabels.data.view_as(predclassrealtrain)).sum()
            totalnblabels += trainreallabels.size(0)
            trainacc = 100 * nbcorrectlabel / totalnblabels
        lossC_CE_real = CELoss(predtrainreallabels, trainreallabels)
        lossC_CE_real.backward(one)

        # train with fake
        #if epoch > 100:
        traingenimages = netG(trainrealimages)
        #filtertraingenimages = netD_Class(traingenimages, onehottrainreallabels).sigmoid()
        #filtertraingenimages = filtertraingenimages.gt(0.3).nonzero()
        # select only gen images seen as belonging to the same class by D
#        print(len(filtertraingenimages))
        #if len(filtertraingenimages) > 0:
        #    selectedtraingenimages = Variable(torch.FloatTensor(len(filtertraingenimages) , traingenimages.size(1), traingenimages.size(2), traingenimages.size(3)))
        #    selectedtraingenlabel = Variable(torch.LongTensor(len(filtertraingenimages)))
        #    if opt.cuda:
        #        selectedtraingenimages, selectedtraingenlabel = selectedtraingenimages.cuda(), selectedtraingenlabel.cuda()
        #    for i in range(len(filtertraingenimages)):
        #        selectedtraingenimages[i, :, :, :] = traingenimages[filtertraingenimages[i].data, :, :, :]
        #        selectedtraingenlabel[i] = trainreallabels[i]
        predtraingenlabels = netC(traingenimages)
            # Compute train loss
        lossC_CE_gen = CELoss(predtraingenlabels, trainreallabels)
        lossC_CE_gen.backward(one)

        #train with unlabeled
        # trainunldata??
        #predunllabels = netC(trainunldata)
        #lossC_unlabeled = netD_Class(trainunldata, predunllabels)
        #lossC_CE_unlabeled.backward(one)

        # Loss
        total_lossC_CE_real += lossC_CE_real.data[0]
        total_lossC_CE_gen += lossC_CE_gen.data[0]
        lossC += total_lossC_CE_real + total_lossC_CE_gen #+ lossC_CE_unlabeled

        optimizerC.step()

        gen_iterations += 1


    if epoch % 10 == 0 and epoch > 0: # print and save loss every 10 epochs

        # Test C
        netC.eval()
        nbcorrectval = 0
        totalval = 0
        for valdata in testloader:
            # Get test images
            valimages, vallabels = valdata
            # activate cuda version of the variables if cuda is activated
            if opt.cuda:
                valimages, vallabels = valimages.cuda(), vallabels.cuda()
            # wrap them in Variable
            valimages, vallabels = Variable(valimages, volatile=True), Variable(vallabels, volatile=True)
            # Calculate scores
            valoutput = netC(valimages)
            # get the index of the max log-probability to get the label
            valpred = valoutput.data.max(1, keepdim=True)[1]
            # count the number of samples correctly classified
            nbcorrectval += valpred.eq(vallabels.data.view_as(valpred)).sum()
            totalval += vallabels.size(0)
            valacc = 100 * nbcorrectval / totalval
            # Compute val loss
            #valloss = CELoss(valoutput, vallabels)

        #if epoch > 100:
            # Print loss on screen for monitoring
        loss = '[{0}/{1}][{2}/{3}][{4}] lossD: {5} lossD_Class: {6} lossD_Dist {7} lossG: {8} lossG_Class: {9} lossG_Dist: {10} lossG_CE: {11} lossC: {12} lossC_CE_real: {13} lossC_CE_gen {14} trainacc {15} valacc {16}'.format(
            epoch, opt.niter, i, len(trainloader), gen_iterations,
            lossD, lossD_Class, lossD_Dist,
            lossG, total_lossG_Class, total_lossG_Dist,total_lossG_CE,
            lossC, total_lossC_CE_real, total_lossC_CE_gen,
            trainacc, valacc)
        print(loss)

        # save loss in a log file
        f = open(opt.outDir + '/' + opt.outDir + '.txt', 'a')
        f.write(loss + '\n')
        f.close
        #else:
            # Print loss on screen for monitoring
        #    loss = '[{0}/{1}][{2}/{3}][{4}] Loss_D: {5} Loss_D_real: {6} Loss_D_aug {7} Loss_G: {8} Loss_G_gen: {9} Loss_G_dist: {10} Loss_G_mutinfo: {11} Loss_C: {12} Loss_C_real: {13} Loss_C_gen 0 Trainacc {14} Valacc {15}'.format(
        #        epoch, opt.niter, i, len(trainloader), gen_iterations,
        #        lossD.data[0], lossD_real.data[0], lossD_gen.data[0],
        #        #lossG.data[0], lossG_gen.data[0], lossG_Dist.data[0], lossG_CE.data[0],
        #        lossG.data[0], lossG_gen.data[0], lossG_Dist.data[0], "0",
        #        lossC.data[0], lossC_CE_real.data[0],
        #        trainacc, valacc)
        #    print(loss)

            # save loss in a log file
        #    f = open(opt.outDir + '/' + opt.outDir + '.txt', 'a')
        #    f.write(loss + '\n')
        #    f.close

    if (epoch % 100 == 0) and (epoch > 0):
        # save some data augmented samples
        #genimages = netG(trainrealimages)
        traingenimages = traingenimages.mul(0.5).add(0.5)
        vutils.save_image(traingenimages.data, '{0}/{1}_generated_samples.png'.format(opt.outDir, gen_iterations))
        reftrainimages = reftrainimages.mul(0.5).add(0.5)
        vutils.save_image(reftrainimages.data, '{0}/{1}_ref_samples.png'.format(opt.outDir, gen_iterations))
        # save some real examples to compare
        realimages = trainrealimages.mul(0.5).add(0.5)
        vutils.save_image(realimages.data, '{0}/{1}_real_samples.png'.format(opt.outDir, gen_iterations))

#    if (epoch % 500 == 0) and (epoch > 0):  # save model every 500 epochs
#        # save final models
#        torch.save(netD_Class.state_dict(), '{0}/netD_Class_epoch_{1}.pth'.format(opt.outDir, epoch))
#        torch.save(netD_Dist.state_dict(), '{0}/netD_Class_epoch_{1}.pth'.format(opt.outDir, epoch))
#        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.outDir, epoch))
#        torch.save(netC.state_dict(), '{0}/netD_Class_epoch_{1}.pth'.format(opt.outDir, epoch))

# save final models
torch.save(netD_Class.state_dict(), '{0}/netD_Class_epoch_{1}.pth'.format(opt.outDir, epoch))
torch.save(netD_Dist.state_dict(), '{0}/netD_Dist_epoch_{1}.pth'.format(opt.outDir, epoch))
torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.outDir, epoch))
torch.save(netC.state_dict(), '{0}/netD_Class_epoch_{1}.pth'.format(opt.outDir, epoch))