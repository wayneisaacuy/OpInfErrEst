#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wayne Isaac Tan Uy, PhD
22 Nov 2020
"""

import numpy as np
import numpy.linalg as LA 
from IntrusiveROM import *

def genAnormBnd(Sys,nTimes,M):
    """
    Generate realization of probabilistic bound for \|A^k\|_2 (AnormBnd)
    without gamma scaling (refer to paper for definition). 
    
    Parameters
    ----------
    Sys : dictionary representing a LTI system.
    nTimes : length of control input.
    M : number of samples over which the maximum is computed.

    Returns
    -------
    AnormBnd : array representing the bound on \|A^k\|_2 for values of k.

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
    
    AnormBnd = np.amax(ThetaNormSamples, axis = 0)
    AnormBnd = AnormBnd.reshape(1,-1)
    
    return AnormBnd

def ReProj(Sys,signal,xInit,V,nSkip):
    """
    Perform re-projection to obtain data for operator inference.

    Parameters
    ----------
    Sys : dictionary representing a LTI system.
    signal : 2-d array for control input.
    xInit : array for initial condition.
    V : basis matrix whose columns are basis vectors.
    nSkip : number of states to skip for re-projection.

    Returns
    -------
    ReProjData: dictionary with keys 'XreProjInit', 'XreProjNextStep', 
                'ureProj', 'residNorm'. 'XreProjNextStep' are the states 
                resulting from performing re-projection for 1 time step at 
                'XreprojInit' with input 'ureProj'. 'residNorm' is the associated
                residual norm.

    """
    A = Sys['A']
    B = Sys['B']
    
    # simulate the full system
    XTraj = TimeStepSys(Sys,signal,xInit)
    
    # pick states at which to perform re-projection for 1 step
    XInit = XTraj[:,:-1:nSkip]
    ureProj = signal[:,::nSkip]
    
    if XInit.shape[1] <= 1:
        raise Exception('Insufficient data for re-projection. Reduce skip size.')
    
    nInit = XInit.shape[1]
    XreProjInit = V.T @ XInit
    XreProjNextStep = np.zeros_like(XreProjInit)
    residNorm = np.zeros((1,nInit))
    
    for k in range(nInit):
        # query system for 1 time step
        xtmp = A @ V @ XreProjInit[:,k:k+1] + B @ ureProj[:,k:k+1]
        # project
        XreProjNextStep[:,k:k+1] = V.T @ xtmp
        # compute residual
        residNorm[:,k:k+1] = LA.norm(xtmp - V @ XreProjNextStep[:,k:k+1])
    
    ReProjData = dict()
    ReProjData['XreProjInit'] = XreProjInit
    ReProjData['XreProjNextStep'] = XreProjNextStep
    ReProjData['ureProj'] = ureProj
    ReProjData['residNorm'] = residNorm 
    
    return ReProjData

def genNonIntROM(ReProjData):
    """
    Perform operator inference to learn the reduced model operators Ar, Br .   

    Parameters
    ----------
    ReProjData : dictionary containing data from the re-projection stage.

    Raises
    ------
    Data matrix has to be full rank.

    Returns
    -------
    NonIntROM : dictionary with keys 'Ar', 'Br'.

    """
    XreProjInit = ReProjData['XreProjInit']
    XreProjNextStep = ReProjData['XreProjNextStep']
    ureProj = ReProjData['ureProj']
    
    # form data matrix
    DataMat = np.concatenate((XreProjInit.T,ureProj.T), axis = 1)
    
    # check if data matrix has enough rows
    
    if DataMat.shape[0] < DataMat.shape[1]:
        raise Exception('Data matrix for operator inference needs more rows.')
    
    # solve least squares problem
    LSSoln, _, rankDataMat, _ = LA.lstsq(DataMat,XreProjNextStep.T)
    
    # check if data matrix is full rank
    
    if rankDataMat < DataMat.shape[1]:
        raise Exception('Data matrix for operator inference is not full rank.')

    rdim = XreProjInit.shape[0]
    ArT = LSSoln[:rdim,:]
    BrT = LSSoln[rdim:,:]
    
    NonIntROM = dict()
    NonIntROM['Ar'] = ArT.T
    NonIntROM['Br'] = BrT.T
    
    return NonIntROM
 
def vech(A):
    """
    Half-vectorization of a symmetric matrix

    Parameters
    ----------
    A : symmetric matrix.

    Returns
    -------
    out : half-vectorization of the symmetric matrix.

    """
    # only for symmetric matrices
    
    dim = A.shape[0]
    idx = np.triu_indices(dim)
    out = A[idx]
    out = out.reshape(out.shape[0],-1)
    
    return out

def genNonIntEE(ReProjData,NonIntROM,V):
    """
    Perform operator inference for the error operators M1, M2, M3.

    Parameters
    ----------
    ReProjData : dictionary containing data from the re-projection stage.
    NonIntROM : dictionary representing the learned reduced system.
    V : basis matrix whose columns are basis vectors.

    Raises
    ------
    Data matrix has to be full rank.

    Returns
    -------
    ErrOp : dictionary with keys 'M1', 'M2', 'M3', 'M4'.

    """
    XreProjInit = ReProjData['XreProjInit']
    XreProjNextStep = ReProjData['XreProjNextStep']
    ureProj = ReProjData['ureProj']
    residNorm = ReProjData['residNorm']
    
    Ar = NonIntROM['Ar']
    Br = NonIntROM['Br']
    
    M4 = V.T @ V
    
    # form right hand side
    
    residNormSq = residNorm ** 2
    rhsMat = np.zeros((residNormSq.shape[1],1))
    
    for k in range(residNormSq.shape[1]):
        currXrNextStep = XreProjNextStep[:,k:k+1]
        currXrInit = XreProjInit[:,k:k+1]
        rhsMat[k:k+1,:] = residNormSq[:,k:k+1] - currXrNextStep.T @ M4 @ currXrNextStep + \
                            2 * currXrNextStep.T @ Ar @ currXrInit + 2 * currXrNextStep.T @ Br @ ureProj[:,k:k+1]
    
    # form the data matrix
    rdim = XreProjInit.shape[0]
    p = ureProj.shape[0]
    DataMat = np.zeros((residNormSq.shape[1],(rdim + p)*(rdim + p + 1)//2))                        
    
    for k in range(residNormSq.shape[1]):
        
        xxT = np.outer(XreProjInit[:,k:k+1], XreProjInit[:,k:k+1])
        tmp = vech(2 * xxT - np.diag(np.diag( xxT )))

        uuT = np.outer(ureProj[:,k:k+1], ureProj[:,k:k+1])   
        tmp = np.vstack((tmp, vech(2 * uuT - np.diag(np.diag( uuT )))))
        
        uxT = ureProj[:,k:k+1] @ XreProjInit[:,k:k+1].T
        tmp = np.vstack((tmp, 2 * uxT.reshape(-1,1)))
        
        DataMat[k:k+1,:] = tmp.T
                        
    # solve the least squares problem
    if DataMat.shape[0] < DataMat.shape[1]:
        raise Exception('Data matrix for error estimation needs more rows.')
    
    LSSoln, _, rankDataMat, _ = LA.lstsq(DataMat,rhsMat)
    
    if rankDataMat < DataMat.shape[1]:
        raise Exception('Data matrix for error estimation is not full rank.')
        
    # extract the operators
    
    M1halfdim = rdim * (rdim + 1) // 2
    M2halfdim = p * (p + 1) //2
    
    M1Vals = LSSoln[:M1halfdim,:]
    M2Vals = LSSoln[M1halfdim:M1halfdim + M2halfdim,:]
    M3Vals = LSSoln[M1halfdim + M2halfdim:,:]
    
    # reshape M1Vals, M2Vals, M3Vals
    
    M1 = np.zeros((rdim,rdim))
    M2 = np.zeros((p,p))
    M3 = M3Vals.reshape(p, rdim)
    
    idM1 = np.triu_indices(rdim)
    idM2 = np.triu_indices(p)
    
    M1[idM1] = M1Vals.reshape(-1,)
    M2[idM2] = M2Vals.reshape(-1,)
    
    M1 = M1 + M1.T - np.diag(np.diag(M1))
    M2 = M2 + M2.T - np.diag(np.diag(M2))
    
    ErrOp = dict()
    ErrOp['M1'] = M1
    ErrOp['M2'] = M2
    ErrOp['M3'] = M3
    ErrOp['M4'] = M4
    
    return ErrOp
        
    