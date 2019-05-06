#Authors: Rajat Sen, Karthikeyan Shanmugam
#Code for project CI testing with generative models. Given a data-set, (X,Y,Z) keep a part of it intact. Use another part of it to train a generative model of Y|Z=z. 
#In a part of the data set replace Y coordinate from generative model (X,Y' = G(Noise, Z),Z). TDo a two-sample test between (X,Y,Z) and (X,Y',Z). 
#Dependencies: CCIT, PyTrorch,numpy,scikit-learn,tensorflow 


import os, sys

sys.path.append(os.getcwd())

import random

#import matplotlib

#import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import tflib as lib
#import tflib.plot

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import copy
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale

from CCIT import *

torch.manual_seed(1)


use_cuda = False
LAMBDA = .1
CRITIC_ITERS = 5
FIXED_GENERATOR = False
DATASET = 'toy'


class Generator(nn.Module):
    '''Input: Noise, Z 
    Fully connected conditional generator'''


    def __init__(self,dim_N,dim_Y,dim_Z,DIM = 15):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(dim_N+dim_Z, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, dim_Z),
            nn.ReLU(True),
            nn.Linear(dim_Z, dim_Z),
            nn.ReLU(True),
            nn.Linear(dim_Z, dim_Y),
        )
        self.main = main

    def forward(self, N , Z):
        I = torch.cat((N,Z),2)
        output = self.main(I)
        return output
    
    
class Discriminator(nn.Module):
    '''Input: Y,Z
    Fully Connected conditional discriminator'''

    def __init__(self,dim_X,dim_Y,dim_Z,DIM = 15):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(dim_X + dim_Y+ dim_Z, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, dim_Z),
            nn.ReLU(True),
            nn.Linear(dim_Z, dim_Z),
            nn.ReLU(True),
            nn.Linear(dim_Z, 1),
        )
        self.main = main

    def forward(self, X, Y, Z):
        I = torch.cat((X,Y,Z),2)
        output = self.main(I)
        return output.view(-1)


class Discriminator_simple(nn.Module):
    '''Input: Y,Z
    Fully Connected conditional discriminator'''

    def __init__(self,dim_X,dim_Y,dim_Z,DIM = 15):
        super(Discriminator_simple, self).__init__()

        main = nn.Sequential(
            nn.Linear(dim_Y+ dim_Z, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, dim_Z),
            nn.ReLU(True),
            nn.Linear(dim_Z, dim_Z),
            nn.ReLU(True),
            nn.Linear(dim_Z, 1),
        )
        self.main = main

    def forward(self, Y, Z):
        I = torch.cat((Y,Z),2)
        output = self.main(I)
        return output.view(-1)


class Discriminator_original(nn.Module):
    '''Input: Y,Z
    Fully Connected conditional discriminator'''

    def __init__(self,dim_X,dim_Y,dim_Z,DIM = 15):
        super(Discriminator_original, self).__init__()

        main = nn.Sequential(
            nn.Linear(dim_Y+ dim_Z, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, dim_Z),
            nn.ReLU(True),
            nn.Linear(dim_Z, dim_Z),
            nn.ReLU(True),
            nn.Linear(dim_Z, 1),
            nn.Sigmoid(),
        )
        self.main = main

    def forward(self, Y, Z):
        I = torch.cat((Y,Z),2)
        output = self.main(I)
        return output.view(-1)



def weights_init(m):
    '''Custom weights initialization (from WGAN pytorch code)'''
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class data_iterator(object):
    '''Data Iterator class for testing and supplying batched data '''
    def __init__(self,dx,dy,dz,sType = 'CI',size = 10000,bsize = 50,nstd = 0.5,fixed_z = False,data = None,channel = 1,normalized = True):
        if data is not None:
            self.dataset = data
        else:
            cov = np.eye(dz)
            mu = np.ones(dz)
            if fixed_z:
                Z = list(np.random.multivariate_normal(mu,cov,1))*size
                Z = np.array(Z)
            else:
                Z = np.random.multivariate_normal(mu,cov,size)
            Z = np.matrix(Z)
            Ax = np.random.rand(dz,dx)
            for i in range(dx):
                Ax[:,i] = Ax[:,i]/np.linalg.norm(Ax[:,i])
            Ax = np.matrix(Ax)
            Ay = np.random.rand(dz,dy)
            for i in range(dy):
                Ay[:,i] = Ay[:,i]/np.linalg.norm(Ay[:,i])
            Ay = np.matrix(Ay)

            Axy = np.random.rand(dx,dy)
            for i in range(dy):
                Axy[:,i] = Axy[:,i]/np.linalg.norm(Axy[:,i])
            Axy = np.matrix(Axy)

            if sType == 'CI':
                X = Z*Ax + nstd*np.random.multivariate_normal(np.zeros(dx),np.eye(dx),size)
                Y = Z*Ay + nstd*np.random.multivariate_normal(np.zeros(dy),np.eye(dy),size)
            elif sType == 'I':
                X = nstd*np.random.multivariate_normal(np.zeros(dx),np.eye(dx),size)
                Y = nstd*np.random.multivariate_normal(np.zeros(dy),np.eye(dy),size)
            else:
                X = np.random.multivariate_normal(np.zeros(dx),np.eye(dx),size)
                Y = 2*X*Axy + Z*Ay + nstd*np.random.multivariate_normal(np.zeros(dy),np.eye(dy),size)
            self.dataset = np.array(np.hstack([X,Y,Z])).astype(np.float32)
        print normalized
        if normalized is True:
            #self.dataset = normalize(self.dataset,axis = 1)
            self.dataset = scale(self.dataset,axis = 0)
            print 'l2 Normalized Dataset'
        s = self.dataset.shape
        self.index = 0
        self.bsize = bsize
        self.size = s[0]
        self.channel = channel
        print 'Initialized Iterator'
        print 'Data Size: ' + str(self.size)
        print 'Batch Size: ' + str(self.bsize)
    def next_batch(self):
        if self.index + self.bsize >= self.size:
            self.index = 0
        i = self.index
        self.index = self.index + self.bsize
        if self.channel > 0:
            return torch.from_numpy(self.dataset[i:i+self.bsize,:]).contiguous().view(self.bsize,self.channel,-1)
        else:
            return torch.from_numpy(self.dataset[i:i+self.bsize,:]).contiguous().view(self.bsize,-1)

def calc_gradient_penalty(netD, real_data_X, real_data_Y, real_data_Z, fake_data,bsize = 50):
    alpha = torch.rand(bsize,1, 1)
    alpha = alpha.expand(real_data_Y.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data_Y + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(Variable(real_data_X,requires_grad=True),interpolates,Variable(real_data_Z,requires_grad=True))

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def calc_gradient_penalty_simple(netD, real_data_X, real_data_Y, real_data_Z, fake_data,bsize = 50):
    alpha = torch.rand(bsize,1, 1)
    alpha = alpha.expand(real_data_Y.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data_Y + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates,Variable(real_data_Z,requires_grad=True))

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def train_conditional_gan(data,dim_N,dim_X,dim_Y,dim_Z, max_epoch, BSIZE,option = 1):
    n = data.shape[0]
    max_iter = max_epoch*n/BSIZE + 1
    print max_iter
    Data = data_iterator(dx=dim_X,dy=dim_Y,dz=dim_Z,sType = 'CI',size = 10000,bsize = BSIZE,nstd = 0.5,fixed_z = False,data = data)
    netG = Generator(dim_N,dim_Y,dim_Z,dim_Z+dim_N)
    netD = Discriminator(dim_X,dim_Y,dim_Z,dim_X + dim_Z+ dim_Y)
    print netG
    print netD
    netD.apply(weights_init)
    netG.apply(weights_init)

    if use_cuda:
        netD = netD.cuda()
        netG = netG.cuda()
    
    optimizerD = optim.Adam(netD.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-3, betas=(0.5, 0.9))

    one = torch.FloatTensor([1])
    mone = one * -1

    if use_cuda:
        one = one.cuda()
        mone = mone.cuda()
    Wloss = []

    for iteration in xrange(max_iter):
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        for iter_d in xrange(CRITIC_ITERS):
            real_data = Data.next_batch()
            if use_cuda:
                real_data = real_data.cuda()
            real_data_X = Variable(real_data[:,:,0:dim_X])
            real_data_Y = Variable(real_data[:,:,dim_X:dim_X+dim_Y])
            real_data_Z = Variable(real_data[:,:,dim_X+dim_Y:dim_X+dim_Y+dim_Z])



            netD.zero_grad()

            # train with real
            D_real = netD(real_data_X,real_data_Y,real_data_Z)
            D_real = D_real.mean()
            D_real.backward(mone)

            # train with fake
            noise = torch.randn(BSIZE,1,dim_N)
            if use_cuda:
                noise = noise.cuda()
            noisev = Variable(noise, volatile=True)  # totally freeze netG
            fake = Variable(netG(noisev, real_data_Z).data)
            inputv = fake
            D_fake = netD(real_data_X, inputv,real_data_Z)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_data_X.data, real_data_Y.data, real_data_Z.data, fake.data,BSIZE)
            gradient_penalty.backward()

            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            Wloss = Wloss + [Wasserstein_D.data.numpy()[0]]
            optimizerD.step()

        if not FIXED_GENERATOR:
            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()

            real_data = Data.next_batch()
            if use_cuda:
                real_data = real_data.cuda()
            real_data_X = Variable(real_data[:,:,0:dim_X])
            real_data_Y = Variable(real_data[:,:,dim_X:dim_X+dim_Y])
            real_data_Z = Variable(real_data[:,:,dim_X+dim_Y:dim_X+dim_Y+dim_Z])

            noise = torch.randn(BSIZE,1, dim_N)
            if use_cuda:
                noise = noise.cuda()
            noisev = Variable(noise)
            #print noisev.size(),real_data_Z.size()
            fake = netG(noisev, real_data_Z)
            G = netD(real_data_X,fake,real_data_Z)
            G = G.mean()
            G.backward(mone)
            G_cost = -G
            optimizerG.step()
            
            if iteration % 100 == 99:
                print 'iter#: ' + str(iteration)
                print 'Wloss:',
                print np.mean(Wloss[-99:])
                
    return netG,netD


def train_conditional_gan_simple(data,dim_N,dim_X,dim_Y,dim_Z, max_epoch, BSIZE,option = 1,normalized = False):
    n = data.shape[0]
    max_iter = max_epoch*n/BSIZE + 1
    print max_iter
    Data = data_iterator(dx=dim_X,dy=dim_Y,dz=dim_Z,sType = 'CI',size = 10000,bsize = BSIZE,nstd = 0.5,fixed_z = False,data = data,normalized=normalized)
    netG = Generator(dim_N,dim_Y,dim_Z,dim_Z+dim_N)
    netD = Discriminator_simple(dim_X,dim_Y,dim_Z,dim_Z+dim_Y)
    print netG
    print netD
    netD.apply(weights_init)
    netG.apply(weights_init)
    
    if use_cuda:
        netD = netD.cuda()
        netG = netG.cuda()
    
    optimizerD = optim.Adam(netD.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-3, betas=(0.5, 0.9))

    one = torch.FloatTensor([1])
    mone = one * -1

    if use_cuda:
        one = one.cuda()
        mone = mone.cuda()
    Wloss = []

    for iteration in xrange(max_iter):
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        for iter_d in xrange(CRITIC_ITERS):
            real_data = Data.next_batch()
            if use_cuda:
                real_data = real_data.cuda()
            real_data_X = Variable(real_data[:,:,0:dim_X])
            real_data_Y = Variable(real_data[:,:,dim_X:dim_X+dim_Y])
            real_data_Z = Variable(real_data[:,:,dim_X+dim_Y:dim_X+dim_Y+dim_Z])

            netD.zero_grad()

            # train with real
            D_real = netD(real_data_Y,real_data_Z)
            D_real = D_real.mean()
            D_real.backward(mone)

            # train with fake
            noise = torch.randn(BSIZE,1,dim_N)
            if use_cuda:
                noise = noise.cuda()
            noisev = Variable(noise, volatile=True)  # totally freeze netG
            fake = Variable(netG(noisev, real_data_Z).data)
            inputv = fake
            D_fake = netD( inputv,real_data_Z)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty_simple(netD, real_data_X.data, real_data_Y.data, real_data_Z.data, fake.data,BSIZE)
            gradient_penalty.backward()

            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            Wloss = Wloss + [Wasserstein_D.data.numpy()[0]]
            optimizerD.step()

        if not FIXED_GENERATOR:
            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()

            real_data = Data.next_batch()
            if use_cuda:
                real_data = real_data.cuda()
            real_data_X = Variable(real_data[:,:,0:dim_X])
            real_data_Y = Variable(real_data[:,:,dim_X:dim_X+dim_Y])
            real_data_Z = Variable(real_data[:,:,dim_X+dim_Y:dim_X+dim_Y+dim_Z])

            noise = torch.randn(BSIZE,1, dim_N)
            if use_cuda:
                noise = noise.cuda()
            noisev = Variable(noise)
            #print noisev.size(),real_data_Z.size()
            fake = netG(noisev, real_data_Z)
            G = netD(fake,real_data_Z)
            G = G.mean()
            G.backward(mone)
            G_cost = -G
            optimizerG.step()
            
            if iteration % 100 == 99:
                print 'iter#: ' + str(iteration)
                print 'Wloss:',
                print np.mean(Wloss[-99:])
                
    if use_cuda:
        return netG.cpu(),netD.cpu()
    return netG,netD


def train_conditional_gan_original(data,dim_N,dim_X,dim_Y,dim_Z, max_epoch, BSIZE,option = 1,normalized = False):
    n = data.shape[0]
    max_iter = max_epoch*n/BSIZE + 1
    print max_iter
    Data = data_iterator(dx=dim_X,dy=dim_Y,dz=dim_Z,sType = 'CI',size = 10000,bsize = BSIZE,nstd = 0.5,fixed_z = False,data = data,normalized=normalized)
    netG = Generator(dim_N,dim_Y,dim_Z,dim_Z+dim_N)
    netD = Discriminator_original(dim_X,dim_Y,dim_Z,dim_Z+dim_Y)
    criterion = nn.BCELoss()
    print netG
    print netD
    netD.apply(weights_init)
    netG.apply(weights_init)
    
    if use_cuda:
        netD = netD.cuda()
        netG = netG.cuda()

    #netG = torch.nn.DataParallel(netG)
    #netD = torch.nn.DataParallel(netD)
    
    optimizerD = optim.Adam(netD.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-3, betas=(0.5, 0.9))

    one = torch.ones((BSIZE))
    mone = torch.zeros((BSIZE))

    if use_cuda:
        one = one.cuda()
        mone = mone.cuda()
    Wloss = []

    for iteration in xrange(max_iter):
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        for iter_d in xrange(CRITIC_ITERS):
            real_data = Data.next_batch()
            if use_cuda:
                real_data = real_data.cuda()
            real_data_X = Variable(real_data[:,:,0:dim_X])
            real_data_Y = Variable(real_data[:,:,dim_X:dim_X+dim_Y])
            real_data_Z = Variable(real_data[:,:,dim_X+dim_Y:dim_X+dim_Y+dim_Z])

            netD.zero_grad()

            # train with real
            D_real = netD(real_data_Y,real_data_Z)
            D_real_error = criterion(D_real,Variable(one))
            D_real_error.backward()
            # train with fake
            noise = torch.randn(BSIZE,1,dim_N)
            if use_cuda:
                noise = noise.cuda()
            noisev = Variable(noise, volatile=True)  # totally freeze netG
            fake = Variable(netG(noisev, real_data_Z).data)
            D_fake = netD( fake,real_data_Z)
            D_fake_error = criterion(D_fake,Variable(mone))
            D_fake_error.backward()
            if use_cuda:
                Wloss = Wloss + [D_fake_error.cpu().data.numpy()[0] + D_real_error.cpu().data.numpy()[0]]
            else:
                Wloss = Wloss + [D_fake_error.data.numpy()[0] + D_real_error.data.numpy()[0]]
            optimizerD.step()

        if not FIXED_GENERATOR:
            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()

            real_data = Data.next_batch()
            if use_cuda:
                real_data = real_data.cuda()
            real_data_X = Variable(real_data[:,:,0:dim_X])
            real_data_Y = Variable(real_data[:,:,dim_X:dim_X+dim_Y])
            real_data_Z = Variable(real_data[:,:,dim_X+dim_Y:dim_X+dim_Y+dim_Z])

            noise = torch.randn(BSIZE,1, dim_N)
            if use_cuda:
                noise = noise.cuda()
            noisev = Variable(noise)
            #print noisev.size(),real_data_Z.size()
            fake = netG(noisev, real_data_Z)
            D_fake = netD( fake ,real_data_Z)
            G_fake_error = criterion(D_fake,Variable(one))
            G_fake_error.backward()
            optimizerG.step()
            
            if iteration % 100 == 99:
                print 'iter#: ' + str(iteration)
                print 'Wloss:',
                print np.mean(Wloss[-99:])
                
    if use_cuda:
        return netG.cpu(),netD.cpu()
    return netG,netD




def pvalue(x,sigma):

    return 0.5*erfc(x/(sigma*np.sqrt(2)))



def CI_sampler_conditional_CGAN(X_in,Y_in,Z_in,train_len = -1,max_epoch=50,BSIZE = 50,option = 1,dim_N = None, normalized = True):
    np.random.seed(11)
    assert (type(X_in) == np.ndarray),"Not an array"
    assert (type(Y_in) == np.ndarray),"Not an array"
    assert (type(Z_in) == np.ndarray),"Not an array"
    
    nx,dx = X_in.shape
    ny,dy = Y_in.shape
    nz,dz = Z_in.shape 

    assert (nx == ny), "Dimension Mismatch"
    assert (nz == ny), "Dimension Mismatch"
    assert (nx == nz), "Dimension Mismatch"

    samples = np.hstack([X_in,Y_in,Z_in]).astype(np.float32)
    if normalized:
        print 'In Sampler Normalized'
        #samples = normalize(samples,axis = 1)
        samples = scale(samples,axis = 0)

    if train_len == -1:
        #train_len = 4*len(X_in)/3
        train_len = 2*len(X_in)/3

    #assert (train_len < 2*nx), "Training length cannot be larger than total length"
    assert (train_len < nx), "Training length cannot be larger than total length"

    data1= samples[0:nx/2,:]
    #data1 = samples
    #data2 = copy.deepcopy(samples)
    data2 = samples[nx/2::,:]
    if dim_N:
        dim_N = dim_N
    else:
        dim_N = dz + 1
    netG, netD = train_conditional_gan_original(data1,dim_N,dx,dy,dz, max_epoch, BSIZE, normalized = normalized)

    n2,m2 = data2.shape
    ntest = Variable(torch.randn(n2,1,dim_N))
    data2_Z = data2[:,dx+dy:dx+dy+dz].reshape((n2,1,dz))
    data2_Z_test = Variable(torch.from_numpy(data2_Z))
    yprime_v = netG(ntest,data2_Z_test)
    yprime = yprime_v.data.numpy().reshape((n2,dy))

    Xset = range(0,dx)
    Yset = range(dx,dx + dy)
    Zset = range(dx + dy,dx + dy + dz)


    X2 = data2[:,Xset]
    Z2 = data2[:,Zset]

    data2_new = np.hstack([X2,yprime,Z2])

    y1 = np.ones([len(data1),1])
    y2 = np.zeros([len(data2_new),1])

    at1 = np.hstack([data1,y1])
    at2 = np.hstack([data2_new,y2])

    if option == 2:
        return at1,at2,netG,netD

    all_train = np.vstack([at1,at2])

    shuffle = np.random.permutation(len(all_train))
    data_final = all_train[shuffle,:]
    l,m = data_final.shape
    Xdata = data_final[:,0:m-1]
    Ydata = data_final[:,m-1]

    Xtrain = Xdata[0:train_len,:]
    Ytrain = Ydata[0:train_len]

    Xtest = Xdata[train_len::,:]
    Ytest = Ydata[train_len::]

    return Xtrain,Ytrain,Xtest,Ytest,netG,netD


def CI_sampler_marginal(X_in,Y_in,Z_in,train_len = -1):
    np.random.seed(11)
    assert (type(X_in) == np.ndarray),"Not an array"
    assert (type(Y_in) == np.ndarray),"Not an array"
    assert (type(Z_in) == np.ndarray),"Not an array"
    
    nx,dx = X_in.shape
    ny,dy = Y_in.shape
    nz,dz = Z_in.shape 

    assert (nx == ny), "Dimension Mismatch"
    assert (nz == ny), "Dimension Mismatch"
    assert (nx == nz), "Dimension Mismatch"

    original_samples = np.hstack([X_in,Y_in,Z_in])

    if train_len == -1:
        #train_len = 4*len(X_in)/3
        train_len = 2*len(X_in)/3

    #assert (train_len < 2*nx), "Training length cannot be larger than total length"
    assert (train_len < nx), "Training length cannot be larger than total length"

    indices = np.random.choice(ny,size=ny,replace=False)

    Yprime = copy.deepcopy(Y_in)
    Yprime = Yprime[indices,:]

    new_samples = np.hstack([X_in,Yprime,Z_in])

    y1 = np.ones([len(original_samples),1])
    y2 = np.zeros([len(new_samples),1])

    at1 = np.hstack([original_samples,y1])
    at2 = np.hstack([new_samples,y2])

    all_train = np.vstack([at1,at2])

    shuffle = np.random.permutation(len(all_train))
    data_final = all_train[shuffle,:]
    l,m = data_final.shape
    Xdata = data_final[:,0:m-1]
    Ydata = data_final[:,m-1]

    Xtrain = Xdata[0:train_len,:]
    Ytrain = Ydata[0:train_len]

    Xtest = Xdata[train_len::,:]
    Ytest = Ydata[train_len::]

    return Xtrain,Ytrain,Xtest,Ytest


def CI_sampler_mixture(X_in,Y_in,Z_in,train_len = -1,max_epoch=50,BSIZE = 50,option = 1,p = 0.3,dim_N = None,normalized = True):
    np.random.seed(11)
    assert (type(X_in) == np.ndarray),"Not an array"
    assert (type(Y_in) == np.ndarray),"Not an array"
    assert (type(Z_in) == np.ndarray),"Not an array"
    
    nx,dx = X_in.shape
    ny,dy = Y_in.shape
    nz,dz = Z_in.shape 

    assert (nx == ny), "Dimension Mismatch"
    assert (nz == ny), "Dimension Mismatch"
    assert (nx == nz), "Dimension Mismatch"

    samples = np.hstack([X_in,Y_in,Z_in]).astype(np.float32)
    if normalized:
        #samples = normalize(samples,axis = 1)
        samples = scale(samples,axis = 0)

    if train_len == -1:
        train_len = 4*len(X_in)/3
        #train_len = 2*len(X_in)/3

    assert (train_len < 2*nx), "Training length cannot be larger than total length"
    #assert (train_len < nx), "Training length cannot be larger than total length"

    #data1= samples[0:nx/2,:]
    data1 = samples
    data2 = copy.deepcopy(samples)
    #data2 = samples[nx/2::,:]
    if dim_N:
        dim_N = dim_N
    else:
        dim_N = dz + 1
    netG, netD = train_conditional_gan_simple(data1,dim_N,dx,dy,dz, max_epoch, BSIZE,normalized = normalized)

    n2,m2 = data2.shape
    ntest = Variable(torch.randn(n2,1,dim_N))
    data2_Z = data2[:,dx+dy:dx+dy+dz].reshape((n2,1,dz))
    data2_Z_test = Variable(torch.from_numpy(data2_Z))
    yprime_v = netG(ntest,data2_Z_test)
    yprime = yprime_v.data.numpy().reshape((n2,dy))

    Xset = range(0,dx)
    Yset = range(dx,dx + dy)
    Zset = range(dx + dy,dx + dy + dz)


    X2 = data2[:,Xset]
    Z2 = data2[:,Zset]
    Y2 = data2[:,Yset]

    yprime2 = copy.deepcopy(Y2)
    k = 1
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree',metric = 'l2').fit(Z2)
    distances, indices = nbrs.kneighbors(Z2)
    for i in range(len(Y2)):
        index = indices[i,k]
        yprime2[i,:] = Y2[index,:]

    indices = np.random.choice(n2,size=int(n2*p),replace=False)

    yprime[indices,:] = yprime2[indices,:]

    data2_new = np.hstack([X2,yprime,Z2])

    y1 = np.ones([len(data1),1])
    y2 = np.zeros([len(data2_new),1])

    at1 = np.hstack([data1,y1])
    at2 = np.hstack([data2_new,y2])

    if option == 2:
        return at1,at2,netG,netD

    all_train = np.vstack([at1,at2])

    shuffle = np.random.permutation(len(all_train))
    data_final = all_train[shuffle,:]
    l,m = data_final.shape
    Xdata = data_final[:,0:m-1]
    Ydata = data_final[:,m-1]

    Xtrain = Xdata[0:train_len,:]
    Ytrain = Ydata[0:train_len]

    Xtest = Xdata[train_len::,:]
    Ytest = Ydata[train_len::]

    return Xtrain,Ytrain,Xtest,Ytest,netG,netD




def CCIT_GAN(X,Y,Z,max_depths = [6,10,13], n_estimators=[100,200,300], colsample_bytrees=[0.8],nfold = 5,feature_selection = 0,train_samp = -1,nthread = 4,max_epoch=100,bsize=50,dim_N = None,normalized = True):
    assert (type(X) == np.ndarray),"Not an array"
    assert (type(Y) == np.ndarray),"Not an array"
    assert (type(Z) == np.ndarray),"Not an array"
    
    nx,dx = X.shape
    ny,dy = Y.shape
    nz,dz = Z.shape 

    assert (nx == ny), "Dimension Mismatch"
    assert (nz == ny), "Dimension Mismatch"
    assert (nx == nz), "Dimension Mismatch"


    np.random.seed(11)
    if train_samp == -1:
        Xtrain,Ytrain,Xtest,Ytest,_,_ = CI_sampler_conditional_CGAN(X,Y,Z,train_len = -1,max_epoch=max_epoch,BSIZE = bsize,dim_N = dim_N,normalized = normalized)
    else:
        Xtrain,Ytrain,Xtest,Ytest,_,_ = CI_sampler_conditional_CGAN(X,Y,Z,train_len = 2*train_samp*nx, max_epoch=max_epoch,BSIZE=bsize,dim_N = dim_N,normalized = normalized)


    model,features,bp = XGB_crossvalidated_model(max_depths=max_depths, n_estimators=n_estimators, colsample_bytrees=colsample_bytrees,Xtrain=Xtrain,Ytrain=Ytrain,nfold=nfold,feature_selection = feature_selection,nthread = nthread)
    gbm = model.fit(Xtrain,Ytrain)
    pred = gbm.predict_proba(Xtest)
    pred_exact = gbm.predict(Xtest)
    acc1 = accuracy_score(Ytest, pred_exact)
    AUC1 = roc_auc_score(Ytest,pred[:,1])
    del gbm
    gbm = model.fit(Xtrain[:,dx::],Ytrain)
    pred = gbm.predict_proba(Xtest[:,dx::])
    pred_exact = gbm.predict(Xtest[:,dx::])
    acc2 = accuracy_score(Ytest, pred_exact)
    AUC2 = roc_auc_score(Ytest,pred[:,1])
    del gbm
    x = [0.0, AUC1 - AUC2 , AUC2 - 0.5, acc1 - acc2, acc2 - 0.5]
    sigma = 1.0/np.sqrt(nx)
    return pvalue(x[1],sigma)


def CCIT_mixture(X,Y,Z,max_depths = [6,10,13], n_estimators=[100,200,300], colsample_bytrees=[0.8],nfold = 5,feature_selection = 0,train_samp = -1,nthread = 4,max_epoch=100,bsize=50):
    assert (type(X) == np.ndarray),"Not an array"
    assert (type(Y) == np.ndarray),"Not an array"
    assert (type(Z) == np.ndarray),"Not an array"
    
    nx,dx = X.shape
    ny,dy = Y.shape
    nz,dz = Z.shape 

    assert (nx == ny), "Dimension Mismatch"
    assert (nz == ny), "Dimension Mismatch"
    assert (nx == nz), "Dimension Mismatch"


    np.random.seed(11)
    if train_samp == -1:
        Xtrain,Ytrain,Xtest,Ytest,_,_ = CI_sampler_mixture(X,Y,Z,train_len = -1,max_epoch=max_epoch,BSIZE = bsize)
    else:
        Xtrain,Ytrain,Xtest,Ytest,_,_ = CI_sampler_mixture(X,Y,Z,train_len = 2*train_samp*nx, max_epoch=max_epoch,BSIZE=bsize)


    model,features,bp = XGB_crossvalidated_model(max_depths=max_depths, n_estimators=n_estimators, colsample_bytrees=colsample_bytrees,Xtrain=Xtrain,Ytrain=Ytrain,nfold=nfold,feature_selection = feature_selection,nthread = nthread)
    gbm = model.fit(Xtrain,Ytrain)
    pred = gbm.predict_proba(Xtest)
    pred_exact = gbm.predict(Xtest)
    acc1 = accuracy_score(Ytest, pred_exact)
    AUC1 = roc_auc_score(Ytest,pred[:,1])
    del gbm
    gbm = model.fit(Xtrain[:,dx::],Ytrain)
    pred = gbm.predict_proba(Xtest[:,dx::])
    pred_exact = gbm.predict(Xtest[:,dx::])
    acc2 = accuracy_score(Ytest, pred_exact)
    AUC2 = roc_auc_score(Ytest,pred[:,1])
    del gbm
    x = [0.0, AUC1 - AUC2 , AUC2 - 0.5, acc1 - acc2, acc2 - 0.5]
    sigma = 1.0/np.sqrt(nx)
    return pvalue(x[1],sigma)


def CCIT_marginal(X,Y,Z,max_depths = [6,10,13], n_estimators=[100,200,300], colsample_bytrees=[0.8],nfold = 5,feature_selection = 0,train_samp = -1,nthread = 4):
    assert (type(X) == np.ndarray),"Not an array"
    assert (type(Y) == np.ndarray),"Not an array"
    assert (type(Z) == np.ndarray),"Not an array"
    
    nx,dx = X.shape
    ny,dy = Y.shape
    nz,dz = Z.shape 

    assert (nx == ny), "Dimension Mismatch"
    assert (nz == ny), "Dimension Mismatch"
    assert (nx == nz), "Dimension Mismatch"


    np.random.seed(11)
    if train_samp == -1:
        Xtrain,Ytrain,Xtest,Ytest= CI_sampler_marginal(X,Y,Z,train_len = -1)
    else:
        Xtrain,Ytrain,Xtest,Ytest = CI_sampler_marginal(X,Y,Z,train_len = train_samp*nx)


    model,features,bp = XGB_crossvalidated_model(max_depths=max_depths, n_estimators=n_estimators, colsample_bytrees=colsample_bytrees,Xtrain=Xtrain,Ytrain=Ytrain,nfold=nfold,feature_selection = feature_selection,nthread = nthread)
    gbm = model.fit(Xtrain,Ytrain)
    pred = gbm.predict_proba(Xtest)
    pred_exact = gbm.predict(Xtest)
    acc1 = accuracy_score(Ytest, pred_exact)
    AUC1 = roc_auc_score(Ytest,pred[:,1])
    del gbm
    gbm = model.fit(Xtrain[:,dx::],Ytrain)
    pred = gbm.predict_proba(Xtest[:,dx::])
    pred_exact = gbm.predict(Xtest[:,dx::])
    acc2 = accuracy_score(Ytest, pred_exact)
    AUC2 = roc_auc_score(Ytest,pred[:,1])
    del gbm
    x = [0.0, AUC1 - AUC2 , AUC2 - 0.5, acc1 - acc2, acc2 - 0.5]
    sigma = 1.0/np.sqrt(nx)
    return pvalue(x[1],sigma)




def DCIT_GAN(X,Y,Z,train_samp = -1,max_epoch=100,bsize=50,n_bootstrap = 20,dim_N = 21):
    assert (type(X) == np.ndarray),"Not an array"
    assert (type(Y) == np.ndarray),"Not an array"
    assert (type(Z) == np.ndarray),"Not an array"
    
    nx,dx = X.shape
    ny,dy = Y.shape
    nz,dz = Z.shape 

    assert (nx == ny), "Dimension Mismatch"
    assert (nz == ny), "Dimension Mismatch"
    assert (nx == nz), "Dimension Mismatch"

    np.random.seed(11)
    if train_samp == -1:
        at1,at2,netG,netD = CI_sampler_conditional_CGAN(X,Y,Z,train_len = -1,max_epoch=max_epoch,BSIZE = bsize,option=2)
    else:
        at1,at2,netG,netD = CI_sampler_conditional_CGAN(X,Y,Z,train_len = train_samp*nx, max_epoch=max_epoch,BSIZE=bsize,option=2)

    n1,m1 = at1.shape
    n2,m2 = at2.shape
    at1_noy = at1[:,0:m1-1].astype(np.float32)
    at2_noy = at2[:,0:m2-1].astype(np.float32)


    at1_v = Variable(torch.from_numpy(at1_noy.reshape(n1,1,m1-1)))
    at2_v = Variable(torch.from_numpy(at2_noy.reshape(n2,1,m2-1)))

    D_real = netD(at1_v[:,:,0:dx],at1_v[:,:,dx:dx+dy],at1_v[:,:,dx+dy:dx+dy+dz])
    D_real = D_real.mean()
    D_fake = netD(at2_v[:,:,0:dx],at2_v[:,:,dx:dx+dy],at2_v[:,:,dx+dy:dx+dy+dz])
    D_fake = D_fake.mean()
    Wloss = D_real - D_fake
    Wloss = Wloss.data.numpy()[0]

    print 'Final Test Wloss:' + str(Wloss)
    
    loss_list = []
    atall = np.vstack([at1_noy,at2_noy])
    n,m = atall.shape
    for i in range(n_bootstrap):
        shuffle1 = np.random.choice(n1,size=n1,replace=True)
        shuffle2 = np.random.choice(n2,size=n2,replace=True)

        a1 = at1_noy[shuffle1,:] #atall[shuffle1,:]
        a2 = at2_noy[shuffle2,:] #atall[shuffle2,:]

        X1 = a1[:,0:dx]
        Z1 = a1[:,dx+dy:dx+dy+dz]
        Z1_prime = Z1.reshape((n1,1,dz))
        
        ntest = Variable(torch.randn(n1,1,dim_N))
        Z1_test = Variable(torch.from_numpy(Z1_prime))
        yprime_v = netG(ntest,Z1_test)
        yprime = yprime_v.data.numpy().reshape((n1,dy))

        at1_prime = np.hstack([X1,yprime,Z1])

        X2 = a2[:,0:dx]
        Z2 = a2[:,dx+dy:dx+dy+dz]
        Z2_prime = Z2.reshape((n/2,1,dz))
        
        ntest = Variable(torch.randn(n2,1,dim_N))
        Z2_test = Variable(torch.from_numpy(Z2_prime))
        yprime_v = netG(ntest,Z2_test)
        yprime = yprime_v.data.numpy().reshape((n2,dy))

        at2_prime = np.hstack([X2,yprime,Z2])

        at1_v = Variable(torch.from_numpy(at1_prime.reshape(n1,1,m)))
        at2_v = Variable(torch.from_numpy(at2_prime.reshape(n2,1,m)))

        D_real = netD(at1_v[:,:,0:dx],at1_v[:,:,dx:dx+dy],at1_v[:,:,dx+dy:dx+dy+dz])
        D_real = D_real.mean()
        D_fake = netD(at2_v[:,:,0:dx],at2_v[:,:,dx:dx+dy],at2_v[:,:,dx+dy:dx+dy+dz])
        D_fake = D_fake.mean()
        loss = D_real - D_fake
        loss = loss.data.numpy()[0]
        loss_list = loss_list + [loss]

    mu = np.mean(loss_list)
    std = np.std(loss_list)
    print 'mean: ' + str(mu)
    print 'std: ' + str(std)

    return pvalue(Wloss - mu,std)





























            
        
    





