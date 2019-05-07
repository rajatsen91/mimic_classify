from __future__ import print_function

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

import xgboost as xgb

#General Headers####################
import numpy as np
import pandas as pd
import random
from multiprocessing import Pool
import copy
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

from math import erfc
import random
#####################################################


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
			print('Std. Normalized Dataset')
		s = self.dataset.shape
		self.index = 0
		self.bsize = bsize
		self.size = s[0]
		self.channel = channel
		print('Initialized Iterator')
		print('Data Size: ' + str(self.size))
		print('Batch Size: ' + str(self.bsize))
	def next_batch(self):
		if self.index + self.bsize >= self.size:
			self.index = 0
		i = self.index
		self.index = self.index + self.bsize
		if self.channel > 0:
			return torch.from_numpy(self.dataset[i:i+self.bsize,:]).contiguous().view(self.bsize,self.channel,-1)
		else:
			return torch.from_numpy(self.dataset[i:i+self.bsize,:]).contiguous().view(self.bsize,-1)


def XGB_crossvalidated_model(max_depths, n_estimators, colsample_bytrees,Xtrain,Ytrain,nfold,feature_selection = 0,nthread = 8):
	'''Function returns a cross-validated hyper parameter tuned model for the training data 
	Arguments:
		max_depths: options for maximum depth eg: input [6,10,13], this will choose the best max_depth among the three
		n_estimators: best number of estimators to be chosen from this. eg: [200,150,100]
		colsample_bytrees: eg. input [0.4,0.8]
		nfold: Number of folds for cross-validated
		Xtrain, Ytrain: Training features and labels
		feature_selection : 0 means feature_selection diabled and 1 otherswise. If 1 then a second output is returned which consists of the selected features

	Output:
		model: Trained model with good hyper-parameters
		features : Coordinates of selected features, if feature_selection = 0
		bp: Dictionary of tuned parameters 

	This procedure is CPU intensive. So, it is advised to not provide too many choices of hyper-parameters
	'''
	classifiers = {}
	model =  xgb.XGBClassifier( nthread=nthread, learning_rate =0.02, n_estimators=100, max_depth=6,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic',scale_pos_weight=1, seed=11)
	model.fit(Xtrain,Ytrain)
	m,n = Xtrain.shape
	features = range(n)
	imp = model.feature_importances_
	if feature_selection == 1:
		features = np.where(imp == 0)[0]
		Xtrain = Xtrain[:,features]
	
	bp = {'max_depth':[0],'n_estimator':[0], 'colsample_bytree' : [0] }
	classifiers['model'] = xgb.XGBClassifier( nthread = nthread, learning_rate =0.02, n_estimators=100, max_depth=6,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.9,objective= 'binary:logistic',scale_pos_weight=1, seed=11)
	classifiers['train_X'] = Xtrain
	classifiers['train_y'] = Ytrain
	maxi = 0
	pos = 0
	for r in max_depths:
		classifiers['model'] = xgb.XGBClassifier( nthread=nthread,learning_rate =0.02, n_estimators=100, max_depth=r,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic',scale_pos_weight=1, seed=11)
		score = cross_validate(classifiers,nfold)
		if maxi < score:
			maxi = score
			pos = r
	bp['max_depth'] = pos
	#print pos
	
	maxi = 0
	pos = 0
	for r in n_estimators:
		classifiers['model'] = xgb.XGBClassifier( nthread=nthread,learning_rate =0.02, n_estimators=r, max_depth=bp['max_depth'],min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic',scale_pos_weight=1, seed=11)
		score = cross_validate(classifiers,nfold)
		if maxi < score:
			maxi = score
			pos = r
	
	bp['n_estimator'] = pos
	#print pos
	
	maxi = 0
	pos = 0
	for r in colsample_bytrees:
		classifiers['model'] = xgb.XGBClassifier( nthread=nthread, learning_rate =0.02, n_estimators=bp['n_estimator'], max_depth=bp['max_depth'],min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=r,objective= 'binary:logistic',scale_pos_weight=1, seed=11)
		score = cross_validate(classifiers,nfold)
		if maxi < score:
			maxi = score
			pos = r
			
	bp['colsample_bytree'] = pos
	model = xgb.XGBClassifier( nthread=nthread,learning_rate =0.02, n_estimators=bp['n_estimator'], max_depth=bp['max_depth'],min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=bp['colsample_bytree'],objective= 'binary:logistic',scale_pos_weight=1, seed=11).fit(Xtrain,Ytrain)
	
	return model,features,bp


def cross_validate(classifier, n_folds = 5):
	'''Custom cross-validation module I always use '''
	train_X = classifier['train_X']
	train_y = classifier['train_y']
	model = classifier['model']
	score = 0.0
	
	skf = KFold(n_splits = n_folds)
	for train_index, test_index in skf.split(train_X):
		X_train, X_test = train_X[train_index], train_X[test_index]
		y_train, y_test = train_y[train_index], train_y[test_index]
		clf = model.fit(X_train,y_train)
		pred = clf.predict_proba(X_test)[:,1]
		#print 'cross', roc_auc_score(y_test,pred)
		score = score + roc_auc_score(y_test,pred)

	return score/n_folds