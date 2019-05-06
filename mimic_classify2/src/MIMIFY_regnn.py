from __future__ import print_function

import sys, os, time


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

from sklearn.neighbors import NearestNeighbors

import xgboost as xgb 





torch.manual_seed(111)
torch.cuda.manual_seed(11)

np.random.seed(111)




use_cuda = False
CRITIC_ITERS = 5
FIXED_GENERATOR = False

from src.utilities import *
from src.CI_base_v2 import *



def CI_sampler_regressor_regnn(X_in,Y_in,Z_in,param_dict):
    #train_len = -1,nthread = 4, max_depth = 6, colsample_bytree = 0.8, n_estimators = 200, noise = 'Normal',perc = 0.3
    train_len = param_dict['train_len']
    nthread = param_dict['nthread']
    max_depth = param_dict['max_depth']
    colsample_bytree = param_dict['colsample_bytree']
    n_estimators = param_dict['n_estimators']
    noise = param_dict['noise']
    normalized = param_dict['normalized']
    print(noise)
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
        train_len = int(2*len(X_in)/3)

    assert (train_len < nx), "Training length cannot be larger than total length"

    data1= samples[0:int(nx/2),:]
    data2 = samples[int(nx/2)::,:]

    multioutputregressor = MultiOutputRegressor(estimator=xgb.XGBRegressor(objective='reg:linear',max_depth=max_depth, \
        colsample_bytree= 1.0, n_estimators=n_estimators,nthread=nthread))

    Xset = range(0,dx)
    Yset = range(dx,dx + dy)
    Zset = range(dx + dy,dx + dy + dz)

    X1,Y1,Z1 = data1[:,Xset],data1[:,Yset],data1[:,Zset]
    X2,Y2,Z2 = data2[:,Xset],data2[:,Yset],data2[:,Zset]    

    Y1prime = copy.deepcopy(Y1)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree',metric = 'l2').fit(Z1)
    distances, indices = nbrs.kneighbors(Z1)
    for i in range(len(Y1)):
        index = indices[i,1]
        Y1prime[i,:] = Y1[index,:]

    MOR = multioutputregressor.fit(Z1,Y1 - Y1prime)


    nz2,mz2 = Z2.shape

    Y2prime = copy.deepcopy(Y2)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree',metric = 'l2').fit(Z2)
    distances, indices = nbrs.kneighbors(Z2)
    for i in range(len(Y2)):
        index = indices[i,1]
        Y2prime[i,:] = Y2[index,:]


    yprime = MOR.predict(Z2) + Y2prime
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



class MIMIFY_regnn(CI_base_v2):
    def __init__(self,X,Y,Z,max_depths = [6,10,13], n_estimators=[100,200,300], colsample_bytrees=[0.8],nfold = 5,\
        train_samp = -1,nthread = 4,max_epoch=100,bsize=50,dim_N = None, noise = 'Laplace',perc = 0.3, normalized = True, deep = False,DIM = 20, \
        deep_classifier = False,params =  {'nhid':20,'nlayers':3}):
        super(MIMIFY_regnn, self).__init__(X,Y,Z,max_depths , n_estimators, colsample_bytrees,nfold,train_samp,nthread ,\
            max_epoch,bsize,dim_N, noise ,perc , normalized, deep_classifier,params)
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
        self.param_dict['DIM'] = DIM
        self.mimic_sampler = CI_sampler_regressor_regnn


