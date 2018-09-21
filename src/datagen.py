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


def same(x):
	return x

def cube(x):
	return np.power(x,3)

def negexp(x):
	return np.exp(-np.abs(x))



def generate_samples_random(size = 1000,sType = 'CI',dx = 1,dy = 1,dz = 20,nstd = 0.5,freq = 1.0, fixed_function = None):
    '''Generate CI,I or NI post-nonlinear samples
    1. Z is independent Gaussian 
    2. X = tanh(<a,Z> + b + noise) and Y = cos(<c,Z> + d + noise) in case of CI
    Arguments:    
        size : number of samples
        sType: CI,I, or NI
        dx: Dimension of X 
        dy: Dimension of Y 
        dz: Dimension of Z 
        nstd: noise standard deviation
        freq: Freq of cosine function
    
    Output:
    	allsamples --> complete data-set
    Note that: 	
    [X = first dx coordinates of allsamples each row is an i.i.d samples]
    [Y = [dx:dx + dy] coordinates of allsamples]
    [Z = [dx+dy:dx+dy+dz] coordinates of all samples]
    '''
    np.random.seed()
    random.seed()
    

    if fixed_function:
        I1 = fixed_function
        I2 = fixed_function
    else:
        I1 = random.randint(1,5)
        I2 = random.randint(1,5)

    if I1 == 1:
    	f1 = same 
    elif I1 == 2:
    	f1 = np.square
    elif I1 == 3:
    	f1 = cube
    elif I1 == 4:
    	f1 = np.tanh
    else:
    	f1 = negexp

    if I2 == 1:
    	f2 = same 
    elif I2 == 2:
    	f2 = np.square
    elif I2 == 3:
    	f2 = cube
    elif I2 == 4:
    	f2 = np.tanh
    else:
    	f2 = negexp
        
    print f1,f2

    num = size
    cov = np.eye(dz)
    mu = np.ones(dz)
    Z = np.random.multivariate_normal(mu,cov,num)
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
        X = f1(freq*(Z*Ax + nstd*np.random.multivariate_normal(np.zeros(dx),np.eye(dx),num)))
        Y = f2(freq*(Z*Ay + nstd*np.random.multivariate_normal(np.zeros(dy),np.eye(dy),num)))
    elif sType == 'I':
        X = f1(freq*(nstd*np.random.multivariate_normal(np.zeros(dx),np.eye(dx),num)))
        Y = f2(freq*(nstd*np.random.multivariate_normal(np.zeros(dy),np.eye(dy),num)))
    else:
        X = f1(freq*(np.random.multivariate_normal(np.zeros(dx),np.eye(dx),num)))
        Y = f2(freq*(2*X*Axy + Z*Ay + nstd*np.random.multivariate_normal(np.zeros(dy),np.eye(dy),num)))
        
    allsamples = np.hstack([X,Y,Z])
    
    return np.array(allsamples)


def random_helper(sims):
	'''
	Helper Function for parallel processing of generate_samples_cos
	'''
	np.random.seed()
	random.seed()
	L = generate_samples_random(size=sims[0],sType=sims[1],dx=sims[2],dy=sims[3],dz=sims[4],nstd=sims[5],freq=sims[6],fixed_function=sims[9])
	s = sims[7] + str(sims[8])+'_'+ str(sims[4]) + '.csv'
	L = pd.DataFrame(L,columns = None)
	L.to_csv(s)
	return 1

#parallel_random_sample_gen(nsamples = 5000,dx = 1,dy = 1,dz = 50,nstd = 0.5,freq = 1,filetype = './data/dim50_random/datafile',num_data = 100, num_proc = 16)

def parallel_random_sample_gen(nsamples = 1000,dx = 1,dy = 1,dz = 20,nstd = 0.5,freq = 1,\
    filetype = '../data/dim20_random/datafile',num_data = 50, num_proc = 4,fixed_function=None):
	''' 
	Function to create several many data-sets of post-nonlinear cos transform half of which are CI and half of which are NI, 
	along wtih the correct labels. The data-sets are stored under a given folder path. 
	############## The path should exist#####################
	For example create a folder ../data/dim20 first. 
	Arguments:
	nsamples: Number of i.i.d samples in each data-set
	dx, dy, dz : Dimension of X, Y, Z
	nstd: Noise Standard Deviation 
	freq: Freq. of cos function 
	filetype: Path to filenames. if filetype = '../data/dim20/datafile', then the files are stored as '.npy' format in folder './dim20' 
	and the files are named datafile0_20.npy .....datafile50_20.npy
	num_data: number of data files 
	num_proc: number of processes to run in parallel 
	
	Output:
	num_data number of datafiles stored in the given folder. 
	datafile.npy files that constains an array that has the correct label. If the first label is '1' then  'datafile20_0.npy' constains a 'CI' dataset. '''
	inputs = []
	stypes = []
	for i in range(num_data):
		x = np.random.binomial(1,0.5)
		if x > 0:
			sType = 'CI'
		else:
			sType = 'NI'
		inputs = inputs + [(nsamples,sType,dx,dy,dz,nstd,freq,filetype,i,fixed_function)]
		stypes = stypes + [x]
	
	np.save(filetype+'.npy',stypes)
	pool = Pool(processes=num_proc)
	result = pool.map(random_helper,inputs)
	cleaned = [x for x in result if not x is None]
	pool.close()

