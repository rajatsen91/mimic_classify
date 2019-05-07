""" Wrapper for the RCIT conditional independence test.
Install the original R package before running this module.
It is available at https://github.com/ericstrobl/RCIT.
Reference:
Strobl, Eric V. and Zhang, Kun and Visweswaran, Shyam,
Approximate Kernel-based Conditional Independence Test for Non-Parametric Causal Discovery,
arXiv preprint arXiv:1202.3775 (2017).
"""

#########################################################

import os
import sys
import numpy as np
import rpy2.robjects as R
from rpy2.robjects.packages import importr
# from utils import np2r
import pdb

importr('RCIT')

def rcit(x, y, z, **kwargs):
    """ Run the RCIT independence test.
    Args:
        x (n_samples, x_dim): First variable.
        y (n_samples, y_dim): Second variable.
        z (n_samples, z_dim): Conditioning variable.
        max_time (float): Time limit for the test -- it will terminate
            after that and return p-value -1.
    Returns:
        p (float): The p-value for the null hypothesis
            that x is independent of y.
    """
    x = np2r(x)
    y = np2r(y)
    z = np2r(z)
    res = R.r.RCIT(x, y, z)
    return res[0][0]

def kcit(x, y, z, **kwargs):
    """ Run the KCIT independence test.
    Args:
        x (n_samples, x_dim): First variable.
        y (n_samples, y_dim): Second variable.
        z (n_samples, z_dim): Conditioning variable.
        max_time (float): Time limit for the test -- it will terminate
            after that and return p-value -1.
    Returns:
        p (float): The p-value for the null hypothesis
            that x is independent of y.
    """
    x = np2r(x)
    y = np2r(y)
    z = np2r(z)
    res = R.r.KCIT(x, y, z)
    return res[0]

def np2r(x):
    """ Convert a numpy array to an R matrix.
    Args:
        x (dim0, dim1): A 2d numpy array.
    Returns:
        x_r: An rpy2 object representing an R matrix isometric to x.
    """
    if 'rpy2' not in sys.modules:
        raise ImportError(("rpy2 is not installed.",
                " Cannot convert a numpy array to an R vector."))
    try:
        dim0, dim1 = x.shape
    except IndexError:
        raise IndexError("Only 2d arrays are supported")
    return R.r.matrix(R.FloatVector(x.flatten()), nrow=dim0, ncol=dim1)
