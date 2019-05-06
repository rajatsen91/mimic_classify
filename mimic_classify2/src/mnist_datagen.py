#General Headers####################
import numpy as np
import pandas as pd
import random
from multiprocessing import Pool
import copy
import random
#####################################

#sklearn headers##################################
from sklearn.metrics import zero_one_loss
import xgboost as xgb
from sklearn import metrics   
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
import itertools
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
#####################################################




import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from PIL import Image


def mnist_datagen(numsamples = 2000, sType = 'CI',dic = None, angle = 20):
    trans = transforms.ToPILImage()
    transT = transforms.ToTensor()
    X = []
    Y = []
    Z = []
    for i in range(numsamples):
        z = np.random.choice(10)
        Z = Z + [z]
        l = len(dic[z])
        xI = np.random.choice(l)
        xT = dic[z][xI]
        X = X + [xT.numpy().reshape(1,28*28)]
        if sType == 'CI':
            yI = np.random.choice(l)
            yT = dic[z][yI]
            yT = transT(trans(yT).rotate(angle))
            Y = Y + [yT.numpy().reshape(1,28*28)]
        else:
            yT = xT
            yT = transT(trans(yT).rotate(angle))
            Y = Y + [yT.numpy().reshape(1,28*28)]
            
    X = np.vstack(X)
    Y = np.vstack(Y)
    Z = np.array(Z,dtype = 'float').reshape(numsamples,1)
    allsamples = np.hstack([X,Y,Z])
    return allsamples


def mnist_helper(sims):
    np.random.seed()
    random.seed()
    samples = mnist_datagen(sims[0],sims[1],sims[2],sims[3])
    s = str(sims[4])+'_'+ str(sims[5]) + '.csv'
    L = pd.DataFrame(samples,columns = None)
    L.to_csv(s)
    return 1

def parallel_mnist_gen(nsamples = 2000, sType = 'CI',dicfile = '../data/mnist_dic.pt', angle = 20, \
    num_proc = 16, num_data = 100, filetype = '../data/mnist/datafile'):
    
    dic = torch.load(dicfile)

    inputs = []
    stypes = []
    for i in range(num_data):
        x = np.random.binomial(1,0.5)
        if x > 0:
            sType = 'CI'
        else:
            sType = 'NI'
        inputs = inputs + [(nsamples,sType,dic,angle,filetype,i)]
        stypes = stypes + [x]
    
    np.save(filetype+'.npy',stypes)
    pool = Pool(processes=num_proc)
    result = pool.map(mnist_helper,inputs)
    cleaned = [x for x in result if not x is None]
    pool.close()



