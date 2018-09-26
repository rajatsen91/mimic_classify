import sys,os
import random
import numpy as np
import sklearn.datasets

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.multioutput import MultiOutputRegressor
import copy
from scipy.stats import laplace

from CCIT import *
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

torch.manual_seed(11)


use_cuda = True
CRITIC_ITERS = 5
FIXED_GENERATOR = False

from utilities import *
from CI_base import *


class Generator(nn.Module):
    '''Input: Noise, Z 
    Fully connected conditional generator'''


    def __init__(self,dim_N,dim_Y,dim_Z,DIM = 15):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(dim_N+dim_Z, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, max(dim_Z,dim_Y)),
            nn.ReLU(True),
            nn.Linear(max(dim_Z,dim_Y), max(dim_Z,dim_Y)),
            nn.ReLU(True),
            nn.Linear(max(dim_Z,dim_Y), dim_Y),
        )
        self.main = main

    def forward(self, N , Z):
        I = torch.cat((N,Z),2)
        output = self.main(I)
        return output


class Discriminator_original(nn.Module):
    '''Input: Y,Z
    Fully Connected conditional discriminator'''

    def __init__(self,dim_X,dim_Y,dim_Z,DIM = 15):
        super(Discriminator_original, self).__init__()

        main = nn.Sequential(
            nn.Linear(dim_Y+ dim_Z, DIM),
            nn.ReLU(True),
            nn.Linear(DIM,max(dim_Z,dim_Y)),
            nn.ReLU(True),
            nn.Linear(max(dim_Z,dim_Y), max(dim_Z,dim_Y)),
            nn.ReLU(True),
            nn.Linear(max(dim_Z,dim_Y), 1),
            nn.Sigmoid(),
        )
        self.main = main

    def forward(self, Y, Z):
        I = torch.cat((Y,Z),2)
        output = self.main(I)
        return output.view(-1)




def train_conditional_gan_original(data,dim_N,dim_X,dim_Y,dim_Z, max_epoch, BSIZE,option = 1,normalized = False):
    n = data.shape[0]
    max_iter = max_epoch*n/BSIZE + 1
    print 'MAX ITER: ' + str(max_iter)
    print 'in train GAN',
    print normalized
    Data = data_iterator(dx=dim_X,dy=dim_Y,dz=dim_Z,bsize = BSIZE,nstd = 0.5,data = data,normalized=normalized)
    netG = Generator(dim_N,dim_Y,dim_Z,dim_Z+dim_N)
    netD = Discriminator_original(dim_X,dim_Y,dim_Z,dim_Z+dim_Y)
    criterion = nn.BCELoss()
    #print netG
    #print netD
    netD.apply(weights_init)
    netG.apply(weights_init)
    if use_cuda:
        netD = netD.cuda()
        netG = netG.cuda()
    optimizerD = optim.Adam(netD.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-3, betas=(0.5, 0.9))

    one = torch.ones((BSIZE))
    mone = torch.zeros((BSIZE))

    if use_cuda:
        one = one.cuda()
        mone = mone.cuda()
    Wloss = []
    print netG, next(netG.parameters()).is_cuda
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
                Wloss = Wloss + [D_fake_error.cpu().data.numpy() + D_real_error.cpu().data.numpy()]
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
                print 'Iter#: ' + str(iteration)
                print 'loss:',
                print np.mean(Wloss[-99:])
    if use_cuda:
        return netG.cpu(),netD.cpu()
    return netG,netD


def CI_sampler_conditional_CGAN(X_in,Y_in,Z_in,param_dict):
    np.random.seed(11)
    nx,dx = X_in.shape
    ny,dy = Y_in.shape
    nz,dz = Z_in.shape
    train_len = param_dict['train_len'] #-1
    max_epoch = param_dict['max_epoch']
    BSIZE = param_dict['BSIZE']
    option = param_dict['option'] #1
    dim_N = param_dict['dim_N']
    normalized = param_dict['normalized']
    print 'in CI Sampler',
    print normalized

    if normalized:
        X_in = scale(X_in,axis = 0,with_mean = False)
        Y_in = scale(Y_in,axis = 0,with_mean = False)
        Z_in = scale(Z_in,axis = 0,with_mean = False)

    samples = np.hstack([X_in,Y_in,Z_in]).astype(np.float32)

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
    netG, netD = train_conditional_gan_original(data1,dim_N,dx,dy,dz, max_epoch, BSIZE,1 , normalized)

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

    return Xtrain,Ytrain,Xtest,Ytest



class MIMIFY_GAN(CI_base):
    def __init__(self,X,Y,Z,max_depths = [6,10,13], n_estimators=[100,200,300], colsample_bytrees=[0.8],nfold = 5,train_samp = -1,nthread = 4,\
        max_epoch=100,bsize=50,dim_N = None, noise = 'Laplace',perc = 0.3, normalized = True,deep_classifier = False,params =  {'nhid':20,'nlayers':3}):
        super(MIMIFY_GAN, self).__init__(X,Y,Z,max_depths , n_estimators, colsample_bytrees,nfold,train_samp,nthread ,max_epoch,bsize,dim_N, noise ,perc , \
            normalized,deep_classifier, params)
        self.param_dict = {}
        self.param_dict['train_len'] = self.train_samp
        self.param_dict['max_epoch'] = self.max_epoch
        self.param_dict['BSIZE'] = self.bsize
        self.param_dict['option'] = 1
        self.param_dict['dim_N'] = self.dim_N
        self.param_dict['normalized'] = self.normalized
        self.mimic_sampler = CI_sampler_conditional_CGAN




