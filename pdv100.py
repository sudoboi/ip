from __future__ import print_function

import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import torchvision

# Specifications
dataroot = "./ip"
workers = 2
batch_size = 1
crop_size = 256
nc = 3
nz = 128
lr = 0.0002
beta1 = 0.5

# Dataset Loader
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.CenterCrop(256),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

#PD-GAN code

def SeqLayer(inc, outc, k, s, p):
    return nn.Sequential(
            nn.Conv2d(inc, outc, 4, 2, 1), #64
            nn.BatchNorm2d(outc),
            nn.ReLU(True)
            )

def GenBlock(inc, outc, stride, padding):
    return nn.Sequential(
            nn.ConvTranspose2d(inc, outc, 4, stride, padding, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(True)
            )

class PriorBlock(nn.Module):
    def __init__(self, outc):
        super(PriorBlock, self).__init__()
        self.outc = outc
        self.priorlayer1 = SeqLayer(3, self.outc, 4, 2, 1)
        self.priorlayer2 = SeqLayer(self.outc*1, self.outc*2, 4, 2, 1) #64
        self.priorlayer3 = SeqLayer(self.outc*2, self.outc*4, 4, 2, 1) #32
        self.priorlayer4 = SeqLayer(self.outc*4, self.outc*8, 4, 2, 1) #16
        self.priorlayer5 = SeqLayer(self.outc*8, self.outc*16, 4, 2, 1) #8
        self.priorlayer6 = SeqLayer(self.outc*16, self.outc*32, 4, 2, 1) #4

    def forward(self, coarse_feats):
        xp1 = self.priorlayer1(coarse_feats)
        xp2 = self.priorlayer2(xp1)
        xp3 = self.priorlayer3(xp2)
        xp4 = self.priorlayer4(xp3)
        xp5 = self.priorlayer5(xp4)
        xp6 = self.priorlayer6(xp5)

        return (xp1, xp2, xp3, xp4, xp5, xp6)
        
# class BetaBlock(nn.Module):
#     def __init__(self, outc):
#         super(BetaBlock, self).__init__()
#         self.outc = outc
#         self.betalayer1 = SeqLayer(self.outc*1, self.outc*2, 4, 2, 1) #64, 256
#         self.betalayer2 = SeqLayer(self.outc*2, self.outc*4, 4, 2, 1) #32, 512
#         self.betalayer3 = SeqLayer(self.outc*4, self.outc*8, 4, 2, 1) #16, 1024
#         self.betalayer4 = SeqLayer(self.outc*8, self.outc*16, 4, 2, 1) #8, 2048
#         self.betalayer5 = SeqLayer(self.outc*16, self.outc*32, 4, 2, 1) #4, 4096

#     def forward(self, coarse_feats):

#         xb1 = self.betalayer1(coarse_feats)
#         xb2 = self.betalayer2(xb1)
#         xb3 = self.betalayer3(xb2)
#         xb4 = self.betalayer4(xb3)
#         xb5 = self.betalayer5(xb4)

#         return (xb1, xb2, xb3, xb4, xb5)

# class GammaBlock(nn.Module):
#     def __init__(self, outc):
#         super(GammaBlock, self).__init__()
#         self.outc = outc
#         self.gammalayer1 = SeqLayer(self.outc*1, self.outc*2, 4, 2, 1) #64, 256
#         self.gammalayer2 = SeqLayer(self.outc*2, self.outc*4, 4, 2, 1) #32, 512
#         self.gammalayer3 = SeqLayer(self.outc*4, self.outc*8, 4, 2, 1) #16, 1024
#         self.gammalayer4 = SeqLayer(self.outc*8, self.outc*16, 4, 2, 1) #8, 2048
#         self.gammalayer5 = SeqLayer(self.outc*16, self.outc*32, 4, 2, 1) #4, 4096

#     def forward(self, coarse_feats):
        
#         xg1 = self.gammalayer1(coarse_feats)
#         xg2 = self.gammalayer2(xg1)
#         xg3 = self.gammalayer3(xg2)
#         xg4 = self.gammalayer4(xg3)
#         xg5 = self.gammalayer5(xg4)

#         return (xg1, xg2, xg3, xg4, xg5)

# Gamma-Beta Block
class GBblock(nn.Module):
    def __init__(self, outc):
        super(GBblock, self).__init__()

        self.outc = outc
        self.gblayer1 = SeqLayer(self.outc*1, self.outc*4, 4, 2, 1)
        self.gblayer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.outc*4, self.outc*8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.outc*8),
            nn.ReLU(True)
        )
        self.gblayer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.outc*8, self.outc*16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.outc*16),
            nn.ReLU(True)
        )
        self.gblayer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.outc*16, self.outc*32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.outc*32),
            nn.ReLU(True)
        )
        self.gblayer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.outc*32, self.outc*64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.outc*64),
            nn.ReLU(True)
        )
    def forward(self, coarse_feats):
        xt1 = self.gblayer1(coarse_feats)
        # print(xt1.shape)
        xb1, xg1 = torch.chunk(xt1, 2, dim = -3)
        xt2 = self.gblayer2(xt1)
        xb2, xg2 = torch.chunk(xt2, 2, dim = -3)
        xt3 = self.gblayer3(xt2)
        xb3, xg3 = torch.chunk(xt3, 2, dim = -3)
        xt4 = self.gblayer4(xt3)
        xb4, xg4 = torch.chunk(xt4, 2, dim = -3)
        xt5 = self.gblayer5(xt4)
        xb5, xg5 = torch.chunk(xt5, 2, dim = -3)

        return (xg1, xg2, xg3, xg4, xg5, xb1, xb2, xb3, xb4, xb5)

class SoftResBlock(nn.Module):

    def __init__(self, inc, idx):
        super(SoftResBlock, self).__init__()
        self.inc = inc
        self.idx = idx

        self.end = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(self.inc, self.inc, 3, 1, 1)
            )
        self.softnorm = nn.Sequential(
            nn.Conv2d(self.inc*2, 1, 3, 1, 1),
            nn.Sigmoid()
            )

    def forward(self, input_feats, prior_feats, beta_feats, gamma_feats, mask):
        
        rsize = 256/pow(2,self.idx+1)
        mask = torch.ones(int(rsize), int(rsize)).to('cuda:0')
        Ds = torch.cat((input_feats, prior_feats), dim=1).to('cuda:0')
        #print(Ds.shape)
        Ds = self.softnorm(Ds)
        #print(Ds.shape)

        Ds = torch.mul(Ds, (1-mask)) + mask

        mu = torch.mean(input_feats, 1, True)
        var = torch.mean(torch.square(input_feats-mu), 1, True)
        eps = 1e-8
        x = (input_feats-mu)/(torch.sqrt(var + eps))
        out = Ds*(gamma_feats*x + beta_feats)
        out = self.end(out)
        return out

class PGenerator(nn.Module):

    def __init__(self, incz, outc):
        super(PGenerator, self).__init__()
        self.incz = incz
        self.outc = outc

        self.reslayer = SeqLayer(3, self.outc, 4, 2, 1) #128, 128
        self.PriorBlock = PriorBlock(self.outc)
        # self.BetaBlock = BetaBlock(self.outc)
        # self.GammaBlock = GammaBlock(self.outc)
        self.GBblock = GBblock(self.outc)

        self.gblock1 = GenBlock(self.incz, self.outc*32, 1, 0) #4*4
        self.SoftResBlock1 = SoftResBlock(self.outc*32, 5)
        self.gblock2 = GenBlock(self.outc*32, self.outc*16, 2, 1) #8*8
        self.SoftResBlock2 = SoftResBlock(self.outc*16, 4)
        self.gblock3 = GenBlock(self.outc*16, self. outc*8, 2, 1) #16*16
        self.SoftResBlock3 = SoftResBlock(self.outc*8, 3)
        self.gblock4 = GenBlock(self.outc*8, self.outc*4, 2, 1) #32*32
        self.SoftResBlock4 = SoftResBlock(self.outc*4, 2)
        self.gblock5 = GenBlock(self.outc*4, self.outc*2, 2, 1) #64*64
        self.SoftResBlock5 = SoftResBlock(self.outc*2, 1)
        self.gblock6 = GenBlock(self.outc*2, self.outc, 2, 1) #128*128
        self.SoftResBlock6 = SoftResBlock(self.outc, 0)
        self.gblock7 = GenBlock(self.outc, 3, 2, 1) #256*256

    def forward(self, input_z, mask, coarse_feats):

        xp1, xp2, xp3, xp4, xp5, xp6 = self.PriorBlock(coarse_feats)
        coarse_feats1 = self.reslayer(coarse_feats)
        # xb1, xb2, xb3, xb4, xb5 = self.BetaBlock(coarse_feats1)
        # xg1, xg2, xg3, xg4, xg5 = self.GammaBlock(coarse_feats1)
        xg1, xg2, xg3, xg4, xg5, xb1, xb2, xb3, xb4, xb5 = self.GBblock(coarse_feats1)

#self, input_feats, prior_feats, beta_feats, gamma_feats, mask

        x = self.gblock1(input_z)
        x = self.SoftResBlock1(x, xp6, xb5, xg5, mask)
        x = self.gblock2(x)
        x = self.SoftResBlock2(x, xp5, xb4, xg4, mask)
        x = self.gblock3(x)
        x = self.SoftResBlock3(x, xp4, xb3, xg3, mask)
        x = self.gblock4(x)
        x = self.SoftResBlock4(x, xp3, xb2, xg2, mask)
        x = self.gblock5(x)
        x = self.SoftResBlock5(x, xp2, xb1, xg1, mask)
        x = self.gblock6(x)
        x = self.SoftResBlock6(x, xp1, coarse_feats1, coarse_feats1, mask)
        x = self.gblock7(x)   

        return x

# PGenerator(incz=128, outc=128)
model = PGenerator(128, 128).to(device)


# VGG Loss
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
            
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


criterion1 = nn.MSELoss()
criterion2 = VGGPerceptualLoss().to(device)
optimizerD = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

num_epochs=200
mask = torch.zeros(1,1, 256, 256)
print("Starting Training...")
# For each epoch
loss = 0
for epoch in range(num_epochs):
    # For each batch in the dataloader
    losst = 0
    print('EPOCH: ' + str(epoch + 1))
    print('===============================================')
    
    for i, data in enumerate(dataloader, 0):
        print(device)
        mask.to(device)
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        
        model.zero_grad()
        noise = torch.randn(1, 128, 1, 1, device=device)
        noise.to(device)
        
        fake = model(noise, mask, inputs).to(device)
        losses = [criterion1(fake, inputs), criterion2(fake, inputs)]
        # losses = [criterion1(fake, inputs)]
        loss = losses[0] * 5 + losses[1]
        print('Loss: ' + str(loss))
        loss.backward()
        optimizerD.step()

        losst += loss
        loss = 0
    if (epoch + 1) % 10 == 0:
      torch.save(model.state_dict(), './checkpoints/' + str(epoch + 1) + '.pt')
    print(losst)


