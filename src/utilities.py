import sys,os
import random
import numpy as np
import sklearn.datasets

from scipy.stats import norm

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


def pvalue(x,sigma):
    return 2.0 - 2.0*norm.cdf(np.abs(x)/sigma)

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
    def __init__(self,dx,dy,dz,sType = 'CI',size = 10000,bsize = 50,nstd = 0.5,fixed_z = False,data = None,channel = 1,normalized = False ):
        if data is not None:
            self.dataset = data
        # else:
        #     cov = np.eye(dz)
        #     mu = np.ones(dz)
        #     if fixed_z:
        #         Z = list(np.random.multivariate_normal(mu,cov,1))*size
        #         Z = np.array(Z)
        #     else:
        #         Z = np.random.multivariate_normal(mu,cov,size)
        #     Z = np.matrix(Z)
        #     Ax = np.random.rand(dz,dx)
        #     for i in range(dx):
        #         Ax[:,i] = Ax[:,i]/np.linalg.norm(Ax[:,i])
        #     Ax = np.matrix(Ax)
        #     Ay = np.random.rand(dz,dy)
        #     for i in range(dy):
        #         Ay[:,i] = Ay[:,i]/np.linalg.norm(Ay[:,i])
        #     Ay = np.matrix(Ay)

        #     Axy = np.random.rand(dx,dy)
        #     for i in range(dy):
        #         Axy[:,i] = Axy[:,i]/np.linalg.norm(Axy[:,i])
        #     Axy = np.matrix(Axy)

        #     if sType == 'CI':
        #         X = Z*Ax + nstd*np.random.multivariate_normal(np.zeros(dx),np.eye(dx),size)
        #         Y = Z*Ay + nstd*np.random.multivariate_normal(np.zeros(dy),np.eye(dy),size)
        #     elif sType == 'I':
        #         X = nstd*np.random.multivariate_normal(np.zeros(dx),np.eye(dx),size)
        #         Y = nstd*np.random.multivariate_normal(np.zeros(dy),np.eye(dy),size)
        #     else:
        #         X = np.random.multivariate_normal(np.zeros(dx),np.eye(dx),size)
        #         Y = 2*X*Axy + Z*Ay + nstd*np.random.multivariate_normal(np.zeros(dy),np.eye(dy),size)
        #     self.dataset = np.array(np.hstack([X,Y,Z])).astype(np.float32)
        #print normalized
        if normalized is True:
            self.dataset = scale(self.dataset,axis = 0,with_mean = False)
            print 'Std. Normalized Dataset'
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