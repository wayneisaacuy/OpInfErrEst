#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wayne Isaac Tan Uy, PhD
22 Nov 2020
"""

import numpy as np
import numpy.linalg as LA
from ROMhelper import *

def genAnormBnd(Sys,nTimes,M):
    """
    Generate realization of probabilistic bound for \|A^k\|_2 (XiSample)
    without gamma scaling (refer to paper for definition). Also compute exact 
    value of \|A^k\|_2 (AtrueNorm) for intrusive approach.

    Parameters
    ----------
    Sys : dictionary representing a LTI system.
    nTimes : length of control input.
    M : number of samples over which the maximum is computed.

    Returns
    -------
    AnormBnd : dictionary with keys XiSample and AtrueNorm.

    """
    
    A = Sys['A']
    B = Sys['B']
    
    p = B.shape[1]
    N = A.shape[0]
    
    # non-intrusive
    
    signal = np.zeros((p,nTimes))
    ThetaNormSamples = np.zeros((M,nTimes))
    
    # Gaussian initial condition
    
    GaussianInit = np.random.normal(0, 1, size = (N, M))
    
    # Query system at Gaussian initial condition and zero input
    
    for k in range(M):
        XTraj = TimeStepSys(Sys, signal, GaussianInit[:,k:k+1])
        XTraj = XTraj[:,1:]
        NormTraj = LA.norm(XTraj, axis = 0)
        ThetaNormSamples[k:k+1,:] = NormTraj
    
    # Compute maximum among samples
    
    XiSample = np.amax(ThetaNormSamples, axis = 0)
    
    # intrusive
    
    AtrueNorm = np.zeros((1,nTimes))
    
    # compute true norm of powers of A
    
    tempA = A
    
    for k in range(nTimes):
        AtrueNorm[:,k:k+1] = LA.norm(tempA, ord = 2)
        tempA = A @ tempA
    
    AnormBnd = dict()
    AnormBnd['XiSample'] = XiSample[None,:]
    AnormBnd['AtrueNorm'] = AtrueNorm
    
    return AnormBnd