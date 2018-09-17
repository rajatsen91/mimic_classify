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


use_cuda = False
CRITIC_ITERS = 5
FIXED_GENERATOR = False

from utilities import *
from CI_base import *

class Regressor(nn.Module):
    '''Input: Noise, Z 
    Fully connected conditional generator'''


    def __init__(self,dim_Y,dim_Z,DIM = 15):
        super(Regressor, self).__init__()

        main = nn.Sequential(
            nn.Linear(dim_Z, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, dim_Z),
            nn.ReLU(True),
            nn.Linear(dim_Z, dim_Z),
            nn.ReLU(True),
            nn.Linear(dim_Z, dim_Y),
        )
        self.main = main

    def forward(self,Z):
        output = self.main(Z)
        return output


def train_regressor(data,dim_X,dim_Y,dim_Z, max_epoch, BSIZE,option = 1):
    n = data.shape[0]
    max_iter = max_epoch*n/BSIZE + 1
    print 'MAX ITER: ' + str(max_iter)
    print 'in train REG',
    Data = data_iterator(dx=dim_X,dy=dim_Y,dz=dim_Z,sType = 'CI',size = 10000,bsize = BSIZE,nstd = 0.5,fixed_z = False,data = data,normalized=False)
    netG = Regressor(dim_Y,dim_Z,dim_Z)
    netG.apply(weights_init)
    criterion = nn.L1Loss()
    
    if use_cuda:
        netG = netG.cuda()

    optimizerG = optim.Adam(netG.parameters(), lr=1e-3, betas=(0.5, 0.9))

    Wloss = []

    for p in netG.parameters():  # reset requires_grad
            p.requires_grad = True 

    for iteration in xrange(max_iter):
        real_data = Data.next_batch()
        if use_cuda:
            real_data = real_data.cuda()
        real_data_X = Variable(real_data[:,:,0:dim_X])
        real_data_Y = Variable(real_data[:,:,dim_X:dim_X+dim_Y])
        real_data_Z = Variable(real_data[:,:,dim_X+dim_Y:dim_X+dim_Y+dim_Z])
        netG.zero_grad()
        fake = netG(real_data_Z)
        G_fake_error = criterion(fake,real_data_Y)
        G_fake_error.backward()
        optimizerG.step()
        Wloss = Wloss + [G_fake_error.data.numpy()[0]]
        if iteration % 100 == 99:
            print 'Iter#: ' + str(iteration)
            print 'loss:',
            print np.mean(Wloss[-99:])
    
    if use_cuda:
        return netG.cpu()
    return netG


def CI_sampler_regressor(X_in,Y_in,Z_in,param_dict):
	#train_len = -1,nthread = 4, max_depth = 6, colsample_bytree = 0.8, n_estimators = 200, noise = 'Normal',perc = 0.3
    train_len = param_dict['train_len']
    nthread = param_dict['nthread']
    max_depth = param_dict['max_depth']
    colsample_bytree = param_dict['colsample_bytree']
    n_estimators = param_dict['n_estimators']
    noise = param_dict['noise']
    normalized = param_dict['normalized']
    print noise
    perc = param_dict['perc']

    if normalized:
    	X_in = scale(X_in,axis = 0,with_mean = False)
    	Y_in = scale(Y_in,axis = 0,with_mean = False)
    	Z_in = scale(Z_in,axis = 0,with_mean = False)


    np.random.seed(11)
    
    nx,dx = X_in.shape
    ny,dy = Y_in.shape
    nz,dz = Z_in.shape 


    samples = np.hstack([X_in,Y_in,Z_in]).astype(np.float32)

    if train_len == -1:
        train_len = 2*len(X_in)/3

    assert (train_len < nx), "Training length cannot be larger than total length"

    data1= samples[0:nx/2,:]
    data2 = samples[nx/2::,:]

    multioutputregressor = MultiOutputRegressor(estimator=xgb.XGBRegressor(objective='reg:linear',max_depth=max_depth, colsample_bytree= 1.0, n_estimators=n_estimators,nthread=nthread))

    Xset = range(0,dx)
    Yset = range(dx,dx + dy)
    Zset = range(dx + dy,dx + dy + dz)

    X1,Y1,Z1 = data1[:,Xset],data1[:,Yset],data1[:,Zset]
    X2,Y2,Z2 = data2[:,Xset],data2[:,Yset],data2[:,Zset]

    if noise == 'Normal':
        MOR = multioutputregressor.fit(Z1,Y1)
        Y1hat = MOR.predict(Z1)
        cov = np.cov(np.transpose(Y1hat - Y1))

        print 'Calculated Covariance: ',
        print cov 

        Yprime = MOR.predict(Z2)
        n2,n22 = data2.shape
        try:
            m1,m2 = cov.shape
            Nprime = np.random.multivariate_normal(np.zeros(m1),cov,size = n2)
        except:
            m1 = 1
            Nprime = np.random.normal(scale=np.sqrt(cov),size = [n2,1])

    elif noise == 'Laplace':
        MOR = multioutputregressor.fit(Z1,Y1)
        Y1hat = MOR.predict(Z1)
        E = Y1 - Y1hat
        Yprime = MOR.predict(Z2)
        n2,n22 = data2.shape
        p,q = E.shape
        s = np.std(E[:,0])
        L = laplace()
        r = L.rvs(size=(n2,1))
        s2 = np.std(r)
        r = (s/s2)*r 
        Nprime = r

        for l in range(1,q):
            s = np.std(E[:,l])
            L = laplace()
            r = L.rvs(size=(n2,1))
            s2 = np.std(r)
            r = (s/s2)*r 
            Nprime = np.vstack((Nprime,r))
    elif noise == 'Mixture':
        MOR = multioutputregressor.fit(Z1,Y1)
        Y1hat = MOR.predict(Z1)
        cov = np.cov(np.transpose(Y1hat - Y1))

        print 'Calculated Covariance: '
        print cov 

        Yprime = MOR.predict(Z2)
        n2,n22 = data2.shape
        try:
            m1,m2 = cov.shape
            Nprime = np.random.multivariate_normal(np.zeros(m1),cov,size = n2)
        except:
            m1 = 1
            NprimeG = np.random.normal(scale=np.sqrt(cov),size = [n2,1])

        MOR = multioutputregressor.fit(Z1,Y1)
        Y1hat = MOR.predict(Z1)
        E = Y1 - Y1hat
        Yprime = MOR.predict(Z2)
        n2,n22 = data2.shape
        p,q = E.shape
        s = np.std(E[:,0])
        L = laplace()
        r = L.rvs(size=(n2,1))
        s2 = np.std(r)
        r = (s/s2)*r 
        Nprime = r

        for l in range(1,q):
            s = np.std(E[:,l])
            L = laplace()
            r = L.rvs(size=(n2,1))
            s2 = np.std(r)
            r = (s/s2)*r 
            Nprime = np.vstack((Nprime,r))
            indices = np.random.choice(p,size=int(perc*p),replace=False)
            Nprime[indices,:] = NprimeG[indices,:]


    else:
        assert False, 'Not Implemented Error'


    yprime = Yprime + Nprime

    data2_new = np.hstack([X2,yprime,Z2])

    y1 = np.ones([len(data1),1])
    y2 = np.zeros([len(data2_new),1])

    at1 = np.hstack([data1,y1])
    at2 = np.hstack([data2_new,y2])


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


def CI_sampler_regressor_v2(X_in,Y_in,Z_in,param_dict):
    #train_len = -1,nthread = 4, max_depth = 6, colsample_bytree = 0.8, n_estimators = 200, noise = 'Normal',perc = 0.3
    train_len = param_dict['train_len']
    nthread = param_dict['nthread']
    max_depth = param_dict['max_depth']
    colsample_bytree = param_dict['colsample_bytree']
    n_estimators = param_dict['n_estimators']
    noise = param_dict['noise']
    normalized = param_dict['normalized']
    print noise
    perc = param_dict['perc']

    if normalized:
        X_in = scale(X_in,axis = 0,with_mean = False)
        Y_in = scale(Y_in,axis = 0,with_mean = False)
        Z_in = scale(Z_in,axis = 0,with_mean = False)


    np.random.seed(11)
    
    nx,dx = X_in.shape
    ny,dy = Y_in.shape
    nz,dz = Z_in.shape 


    samples = np.hstack([X_in,Y_in,Z_in]).astype(np.float32)

    if train_len == -1:
        train_len = 2*len(X_in)/3

    assert (train_len < nx), "Training length cannot be larger than total length"

    data1= samples[0:nx/2,:]
    data2 = samples[nx/2::,:]

    multioutputregressor = MultiOutputRegressor(estimator=xgb.XGBRegressor(objective='reg:linear',max_depth=max_depth, colsample_bytree= 1.0, n_estimators=n_estimators,nthread=nthread))

    Xset = range(0,dx)
    Yset = range(dx,dx + dy)
    Zset = range(dx + dy,dx + dy + dz)

    X1,Y1,Z1 = data1[:,Xset],data1[:,Yset],data1[:,Zset]
    X2,Y2,Z2 = data2[:,Xset],data2[:,Yset],data2[:,Zset]

    MOR = multioutputregressor.fit(Z1,Y1)
    Y1hat = MOR.predict(Z1)
    cov = np.cov(np.transpose(Y1hat - Y1))

    print 'Calculated Covariance: ',
    print cov 

    sigmas = [2**l for l in range(-10,4)]
    nz1,mz1 = Z1.shape

    mini = 1e6
    zsigma = 0 

    for s in sigmas:
        znoise = np.random.normal(scale = s,size = [nz1,mz1])
        Zprime = Z1  + znoise
        Y1hatprime = MOR.predict(Zprime)
        cov2 = np.cov(np.transpose(Y1hat - Y1hatprime))
        mae = np.abs(cov2 - cov).mean()
        if mae <= mini:
            mini = mae
            zsigma = s

    print 'Selected Zsigma: ' + str(zsigma) 


    nz2,mz2 = Z2.shape
    yprime = MOR.predict(Z2 + np.random.normal(scale = zsigma,size = [nz2,mz2]))
    n2,n22 = data2.shape


    data2_new = np.hstack([X2,yprime,Z2])

    y1 = np.ones([len(data1),1])
    y2 = np.zeros([len(data2_new),1])

    at1 = np.hstack([data1,y1])
    at2 = np.hstack([data2_new,y2])


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


def CI_sampler_regressor_v3(X_in,Y_in,Z_in,param_dict):
    #train_len = -1,nthread = 4, max_depth = 6, colsample_bytree = 0.8, n_estimators = 200, noise = 'Normal',perc = 0.3
    train_len = param_dict['train_len']
    nthread = param_dict['nthread']
    max_depth = param_dict['max_depth']
    colsample_bytree = param_dict['colsample_bytree']
    n_estimators = param_dict['n_estimators']
    noise = param_dict['noise']
    normalized = param_dict['normalized']
    max_epoch = param_dict['max_epoch']
    BSIZE = param_dict['BSIZE']
    print noise
    perc = param_dict['perc']

    if normalized:
        X_in = scale(X_in,axis = 0,with_mean = False)
        Y_in = scale(Y_in,axis = 0,with_mean = False)
        Z_in = scale(Z_in,axis = 0,with_mean = False)


    np.random.seed(11)
    
    nx,dx = X_in.shape
    ny,dy = Y_in.shape
    nz,dz = Z_in.shape 


    samples = np.hstack([X_in,Y_in,Z_in]).astype(np.float32)

    if train_len == -1:
        train_len = 2*len(X_in)/3

    assert (train_len < nx), "Training length cannot be larger than total length"

    data1= samples[0:nx/2,:]
    data2 = samples[nx/2::,:]

    netG = train_regressor(data1,dx,dy,dz, max_epoch=max_epoch, BSIZE=BSIZE,option = 1)

    Xset = range(0,dx)
    Yset = range(dx,dx + dy)
    Zset = range(dx + dy,dx + dy + dz)

    X1,Y1,Z1 = data1[:,Xset],data1[:,Yset],data1[:,Zset]
    X2,Y2,Z2 = data2[:,Xset],data2[:,Yset],data2[:,Zset]

    nz1,mz1 = Z1.shape
    Z1T = Variable(torch.from_numpy(Z1.reshape(nz1,1,dz)).float())
    Y1hatT = netG(Z1T)
    Y1hat = Y1hatT.data.numpy().reshape(Y1.shape)
    cov = np.cov(np.transpose(Y1hat - Y1))

    print 'Calculated Covariance: ',
    print cov 

    sigmas = [2**l for l in range(-10,4)]
    

    mini = 1e6
    zsigma = 0 

    for s in sigmas:
        znoise = np.random.normal(scale = s,size = [nz1,mz1])
        Zprime = Z1  + znoise
        nz1,mz1 = Z1.shape
        Z1T = Variable(torch.from_numpy(Zprime.reshape(nz1,1,dz)).float())
        Y1hatT = netG(Z1T)
        Y1hatprime = Y1hatT.data.numpy().reshape(Y1.shape)
        cov2 = np.cov(np.transpose(Y1hat - Y1hatprime))
        mae = np.abs(cov2 - cov).mean()
        if mae <= mini:
            mini = mae
            zsigma = s

    print 'Selected Zsigma: ' + str(zsigma) 


    nz2,mz2 = Z2.shape
    Z2prime = Z2 + np.random.normal(scale = zsigma,size = [nz2,mz2])
    Z2T = Variable(torch.from_numpy(Z2prime.reshape(nz2,1,dz)).float())
    Y2hatT = netG(Z2T)
    yprime = Y2hatT.data.numpy().reshape(Y2.shape)
    n2,n22 = data2.shape


    data2_new = np.hstack([X2,yprime,Z2])

    y1 = np.ones([len(data1),1])
    y2 = np.zeros([len(data2_new),1])

    at1 = np.hstack([data1,y1])
    at2 = np.hstack([data2_new,y2])


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


class MIMIFY_REG(CI_base):
    def __init__(self,X,Y,Z,max_depths = [6,10,13], n_estimators=[100,200,300], colsample_bytrees=[0.8],nfold = 5,train_samp = -1,nthread = 4,max_epoch=100,bsize=50,dim_N = None, noise = 'Laplace',perc = 0.3, normalized = True, deep = False):
        super(MIMIFY_REG, self).__init__(X,Y,Z,max_depths , n_estimators, colsample_bytrees,nfold,train_samp,nthread ,max_epoch,bsize,dim_N, noise ,perc , normalized )
        self.param_dict = {}
        self.param_dict['train_len'] = self.train_samp
        self.param_dict['max_depth'] = self.max_depths[0]
        self.param_dict['colsample_bytree'] = self.colsample_bytrees[0]
        self.param_dict['n_estimators'] = self.n_estimators[1]
        self.param_dict['noise'] = self.noise
        self.param_dict['perc'] = self.perc
        self.param_dict['nthread'] = self.nthread
        self.param_dict['normalized'] = self.normalized
        self.param_dict['BSIZE'] = self.bsize
        self.param_dict['max_epoch'] = self.max_epoch
        if deep:
            self.mimic_sampler = CI_sampler_regressor_v3
        else:
            self.mimic_sampler = CI_sampler_regressor_v2


