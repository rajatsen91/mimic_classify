from CCIT import *
import numpy as np
import pandas as pd
from src.MIMIFY_GAN import *
from src.MIMIFY_REG import *

import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description='runs CI test from a folder')
    # Add arguments
    parser.add_argument(
        '-fp', '--folder_path', type=str, help='Data Folder Path', required=True)
    parser.add_argument(
        '-dx', '--dimX', type=int, help='dimension of X', required=True)
    parser.add_argument(
        '-dy', '--dimY', type=int, help='dimension of Y', required=True)
    parser.add_argument(
        '-dz', '--dimZ', type=int, help='dimension of X', required=True)
    parser.add_argument(
        '-dn', '--dimN', type=int, help='dimension of noise', required=False)
    parser.add_argument(
        '-nf', '--numfile', type=int, help='number of files', required=True)
    parser.add_argument(
        '-nt', '--nthread', type=int, help='number of threads', required=False)
    parser.add_argument(
        '-me', '--maxepoch', type=int, help='max. number of epochs to train regressor', required=False)
    parser.add_argument(
        '-bs', '--bsize', type=int, help='batch size', required=False)
    parser.add_argument(
        '-rf', '--rfile', type=str, help='result file name', required=True)

    # Array for all arguments passed to script
    args = parser.parse_args()
    folder_path = args.folder_path
    dx = args.dimX
    dy = args.dimY
    dz = args.dimZ
    numfile = args.numfile
    if args.nthread:
        nthread = args.nthread
    else:
        nthread = 16

    if args.dimN:
        dimN = args.dimN 
    else:
        dimN = 30

    if args.maxepoch:
        maxepoch = args.maxepoch
    else:
        maxepoch = 50
    if args.bsize:
        bsize = args.bsize
    else:
        bsize = 50
    rfile = args.rfile

    return folder_path,dx,dy,dz,numfile,nthread,dimN,maxepoch,bsize,rfile




if __name__ == "__main__":
    folder_path,dx,dy,dz,numfile,nthread,dim_N,maxepoch,bsize,rfile = get_args()
    
    pvalues = []
    d = np.load(folder_path + 'datafile.npy')
    for i in range(numfile):
        print '#iter: ' + str(i)
        datafile = folder_path  + 'datafile' + str(i) + '_' + str(dz) + '.csv'
        y = pd.read_csv(datafile,header = None)
        y = np.array(y).astype(np.float32)
        y = y[1::,1::]
        
        MCG = MIMIFY_GAN(y[:,0:dx],y[:,dx:dx+dy],y[:,dx+dy:dx+dy+dz],\
                 normalized=False,nthread = nthread, dim_N = dim_N, max_epoch = maxepoch, bsize = bsize)
        pvalues = pvalues + [MCG.CI_classify()] 
        print i,d[i],pvalues[-1]

        np.save(folder_path + rfile + '.npy' ,pvalues)

