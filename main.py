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
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='final number of filters generator')
parser.add_argument('--ndf', type=int, default=64, help='initial number of filters discriminator')
parser.add_argument('--ncf', type=int, default=64, help='initial number of filters classifier')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--lrC', type=float, default=0.00005, help='learning rate for Classifier, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netDClass', default='', help="path to netDClass (to continue training)")
parser.add_argument('--netDDist', default='', help="path to netDDist (to continue training)")
parser.add_argument('--netC', default='', help="path to netC (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.05)
parser.add_argument('--clamp_upper', type=float, default=0.05)
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

# activate cuda acceleration if available
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# activate cuda acceleration if available
device = torch.device("cuda" if opt.cuda else "cpu")

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

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
ncf = int(opt.ncf)
nc = int(opt.nc)

# custom weights initialization called on netG and netDClass
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
netDClass = discriminator_model.DClass(opt.imageSize, nc, ndf, nclasses).to(device)
netDClass.apply(weights_init)
netDDist = discriminator_model.DDist(opt.imageSize, nc, ndf).to(device)
netDDist.apply(weights_init)
# Generator
netG = generator_model.UNet(nc,nc,nz).to(device)
netG.apply(weights_init)
# Classifier
netC = classifier_model.cnnClass(nc, ncf).to(device)
netC.apply(weights_init)

# Load model checkpoint if needed
if opt.netDClass != '':
    netDClass.load_state_dict(torch.load(opt.netDClass))
print(netDClass)
if opt.netDDist != '':
    netDDist.load_state_dict(torch.load(opt.netDDist))
print(netDDist)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)
if opt.netC != '':
    netDClass.load_state_dict(torch.load(opt.netC))
print(netC)

# define helpers for optimisation
one = torch.FloatTensor([1]).to(device)
mone = one * -1

# setup optimizer
if opt.adam:
    optimizerDClass = optim.Adam(netDClass.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerDDist = optim.Adam(netDDist.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    optimizerC = optim.Adam(netC.parameters(), lr=opt.lrC, betas=(opt.beta1, 0.999))
else:
    optimizerDClass = optim.RMSprop(netDClass.parameters(), lr=opt.lrD)
    optimizerDDist = optim.RMSprop(netDDist.parameters(), lr=opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)
    optimizerC = optim.RMSprop(netC.parameters(), lr=opt.lrC)

# define classification loss
CELoss = nn.CrossEntropyLoss().to(device)
if opt.kldiv:
    BCELoss = nn.BCEWithLogitsLoss().to(device)

gen_iterations = 0
for epoch in range(opt.niter):
    data_iter = iter(trainloader)
    i = 0

    # Initialize losses
    lossD = 0.0
    lossDClass = 0.0
    total_lossDClass_real = 0.0
    total_lossDClass_gen = 0.0
    lossDDist = 0.0
    total_lossDDist_real = 0.0
    total_lossDDist_gen = 0.0
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
        for p in netDClass.parameters():
            p.requires_grad = True
        for p in netDDist.parameters():
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

            # clamp parameters to a cube if using Wasserstein distance optimisation
            if not opt.kldiv:
                for p in netDClass.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            data = data_iter.next()

            # Sample batch of real training images
            trainrealimages, trainreallabels = data
            batch_size = trainrealimages.size(0)
            trainrealimages =  trainrealimages.requires_grad_().to(device)
            trainreallabels = trainreallabels.requires_grad_().to(device)
            noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).requires_grad_().to(device)
            noise.normal_(0, 1)

            if opt.kldiv:
                label_1 = torch.FloatTensor(batch_size).fill_(1).to(device)
                #label_1.requires_grad_()
            # converting labels to one hot encoding form
            onehotlabelssupport = torch.FloatTensor(batch_size, nclasses).zero_().to(device)

            onehottrainreallabels = onehotlabelssupport.scatter_(1,trainreallabels.unsqueeze(1), 1)

            ##############################
            # (1.1) Update DClass network
            ##############################
            netDClass.zero_grad()

            output_1 = netDClass(trainrealimages, onehottrainreallabels)
            # train with real
            if opt.kldiv:
                lossDClass_real = BCELoss(output_1, label_1)
            else:
                lossDClass_real = output_1.mean()
            lossDClass_real.backward(mone)

            # train with generated
            traingenimages = netG(trainrealimages, noise)
            output_0 = netDClass(traingenimages, onehottrainreallabels)
            if opt.kldiv:
                lossDClass_gen = BCELoss(output_0, label_1)
            else:
                lossDClass_gen = output_0.mean()
            lossDClass_gen.backward(one, retain_graph=True)

            # train with unlabeled
            # complete with unlabeled label

            #lossD_unlabeled = netDClass(traingenimages, trainreallabels)
            #lossD_unlabeled.backward(mone)

            total_lossDClass_real += - lossDClass_real.item()
            total_lossDClass_gen += lossDClass_gen.item()
            lossDClass += total_lossDClass_real + total_lossDClass_gen #- lossD_unlabeled
            optimizerDClass.step()

            #############################
            # (1.2) Update DDist network
            #############################
            netDDist.zero_grad()

            # Minimize distance between input sample and a sample from same class
            reftrainimages = trainrealimages.clone()
            for i in range(reftrainimages.size(0)):
                found = False
                while not found:
                    index = random.randint(0, len(trainset)-1)
                    selected_image, selected_label = trainset.__getitem__(index)
                    selected_image = selected_image.to(device)
                    selected_image.requires_grad_()
                    if selected_label != trainreallabels.data[i]:
                        found = True
                        reftrainimages[i] = selected_image

            output_dist_1 = netDDist(trainrealimages, reftrainimages)
            if opt.kldiv:
                lossDDist_real = BCELoss(output_dist_1, label_1)
            else:
                lossDDist_real = output_dist_1.mean()
            lossDDist_real.backward(mone, retain_graph=True)
            # Maximize distance between input sample and generated sample
            output_dist_0 = netDDist(trainrealimages, traingenimages)
            if opt.kldiv:
                lossDDist_gen = BCELoss(output_dist_0, label_1)
            else:
                lossDDist_gen = output_dist_0.mean()
            lossDDist_gen.backward(one)

            # Loss
            total_lossDDist_real += - lossDDist_real.item()
            total_lossDDist_gen += lossDDist_gen.item()
            lossDDist += total_lossDDist_real + total_lossDDist_gen
            optimizerDDist.step()

            lossD += lossDClass + lossDDist

        ############################
        # (2) Update G network
        ###########################
        for p in netDDist.parameters():
            p.requires_grad = False # to avoid computation
        for p in netDClass.parameters():
            p.requires_grad = False # to avoid computation
        for p in netG.parameters():
            p.requires_grad = True
        for p in netC.parameters():
            p.requires_grad = False # to avoid computation

        netG.zero_grad()
        traingenimages = netG(trainrealimages, noise)
        output_0 = netDClass(traingenimages, onehottrainreallabels)
        # True/Fake Loss
        if opt.kldiv:
           lossG_Class = BCELoss(output_0, label_1)
        else:
           lossG_Class = output_0.mean()
        lossG_Class.backward(mone, retain_graph=True)
        # Distance loss
        output_1 = netDDist(trainrealimages, traingenimages)
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
        total_lossG_Class += - lossG_Class.item()
        total_lossG_Dist += - lossG_Dist.item()
        total_lossG_CE += lossG_CE.item()
        lossG += total_lossG_Class + total_lossG_Dist + total_lossG_CE

        optimizerG.step()

        ############################
        # (3) Update C network
        ###########################
        for p in netDDist.parameters():
            p.requires_grad = False # to avoid computation
        for p in netDClass.parameters():
            p.requires_grad = False # to avoid computation
        for p in netG.parameters():
            p.requires_grad = False # to avoid computation
        for p in netC.parameters():
            p.requires_grad = True

        netC.train()
        netC.zero_grad()

        # train with real
        predtrainreallabels = netC(trainrealimages)
        nbcorrectlabel = 0.0
        totalnblabels = 0.0
        if epoch % 10 == 0 and epoch > 0:  # print and save loss every 10 epochs
            with torch.no_grad():
                # get the index of the max log-probability to get the label
                predclassrealtrain = predtrainreallabels.data.max(1, keepdim=True)[1]
                # count the number of samples correctly classified
                nbcorrectlabel += predclassrealtrain.eq(trainreallabels.data.view_as(predclassrealtrain)).sum()
                totalnblabels += trainreallabels.size(0)
                trainacc = 100 * nbcorrectlabel.item() / totalnblabels
        lossC_CE_real = CELoss(predtrainreallabels, trainreallabels)
        lossC_CE_real.backward(one)

        # train with fake
        traingenimages = netG(trainrealimages, noise)
        predtraingenlabels = netC(traingenimages)
        # Compute train loss
        lossC_CE_gen = CELoss(predtraingenlabels, trainreallabels)
        lossC_CE_gen.backward(one)

        #train with unlabeled
        # trainunldata??
        #predunllabels = netC(trainunldata)
        #lossC_unlabeled = netDClass(trainunldata, predunllabels)
        #lossC_CE_unlabeled.backward(one)

        # Loss
        total_lossC_CE_real += lossC_CE_real.item()
        total_lossC_CE_gen += lossC_CE_gen.item()
        lossC += total_lossC_CE_real + total_lossC_CE_gen #+ lossC_CE_unlabeled

        optimizerC.step()

        gen_iterations += 1


    if epoch % 10 == 0 and epoch > 0: # print and save loss every 10 epochs

        # Test C
        netC.eval()
        nbcorrectval = 0.0
        totalval = 0.0
        with torch.no_grad():
            for valdata in testloader:
                # Get test images
                valimages, vallabels = valdata
                # activate cuda version of the variables if cuda is activated
#                if opt.cuda:
                valimages, vallabels = valimages.to(device), vallabels.to(device)
                # wrap them in Variable
#                valimages, vallabels = Variable(valimages, volatile=True), Variable(vallabels, volatile=True)
                # Calculate scores
                valoutput = netC(valimages)
                # get the index of the max log-probability to get the label
                valpred = valoutput.data.max(1, keepdim=True)[1]
                # count the number of samples correctly classified
                nbcorrectval += valpred.eq(vallabels.view_as(valpred)).sum()
                totalval += vallabels.size(0)
                valacc = 100 * nbcorrectval.item() / totalval
                # Compute val loss
                #valloss = CELoss(valoutput, vallabels)

        #if epoch > 100:
            # Print loss on screen for monitoring
        loss = '[{0}/{1}][{2}/{3}][{4}] lossD: {5} lossDClass: {6} lossDDist {7} lossG: {8} lossG_Class: {9} lossG_Dist: {10} lossG_CE: {11} lossC: {12} lossC_CE_real: {13} lossC_CE_gen {14} trainacc {15} valacc {16}'.format(
            epoch, opt.niter, i, len(trainloader), gen_iterations,
            lossD, lossDClass, lossDDist,
            lossG, total_lossG_Class, total_lossG_Dist,total_lossG_CE,
            lossC, total_lossC_CE_real, total_lossC_CE_gen,
            trainacc, valacc)
        print(loss)

        # save loss in a log file
        f = open(opt.outDir + '/' + opt.outDir + '.txt', 'a')
        f.write(loss + '\n')
        f.close

    if (epoch % 100 == 0) and (epoch > 0):
        # save some examples
        traingenimages = traingenimages.mul(0.5).add(0.5)
        vutils.save_image(traingenimages.data, '{0}/{1}_generated_samples.png'.format(opt.outDir, gen_iterations))
        reftrainimages = reftrainimages.mul(0.5).add(0.5)
        vutils.save_image(reftrainimages.data, '{0}/{1}_ref_samples.png'.format(opt.outDir, gen_iterations))
        realimages = trainrealimages.mul(0.5).add(0.5)
        vutils.save_image(realimages.data, '{0}/{1}_real_samples.png'.format(opt.outDir, gen_iterations))

#    if (epoch % 500 == 0) and (epoch > 0):  # save model every 500 epochs
#        # save final models
#        torch.save(netDClass.state_dict(), '{0}/netDClass_epoch_{1}.pth'.format(opt.outDir, epoch))
#        torch.save(netDDist.state_dict(), '{0}/netDClass_epoch_{1}.pth'.format(opt.outDir, epoch))
#        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.outDir, epoch))
#        torch.save(netC.state_dict(), '{0}/netDClass_epoch_{1}.pth'.format(opt.outDir, epoch))

# save final models
torch.save(netDClass.state_dict(), '{0}/netDClass_epoch_{1}.pth'.format(opt.outDir, epoch))
torch.save(netDDist.state_dict(), '{0}/netDDist_epoch_{1}.pth'.format(opt.outDir, epoch))
torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.outDir, epoch))
torch.save(netC.state_dict(), '{0}/netDClass_epoch_{1}.pth'.format(opt.outDir, epoch))