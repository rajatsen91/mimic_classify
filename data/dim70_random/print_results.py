import numpy as np 
import pandas as pd 
from sklearn.metrics import roc_auc_score 








y = np.load('datafile.npy')

try:
	K = pd.read_csv('KCIT.csv') 
	xrk = np.array(K['b'])
	print 'KCIT: ' + str(roc_auc_score(y,xrk))
except:
	print 'KCIT not completed'

try:
	R = pd.read_csv('RCIT.csv') 
	xrc = np.array(R['b'])
	print 'RCIT: ' + str(roc_auc_score(y,xrc))
except:
	print 'RCIT not completed'

try:
	xr = np.load('result_ccit_regressor_mixed.npy')
	print 'regCIT: ' + str(roc_auc_score(y,xr))
except:
	print 'regCIT not completed'

try:
	xc = np.load('result_ccit.npy')
	print 'CCIT: ' + str(roc_auc_score(y,xc))
except:
	print 'CCIT not completed'

