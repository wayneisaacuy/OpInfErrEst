#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wayne Isaac Tan Uy, PhD
22 Nov 2020
"""

import numpy as np
import numpy.linalg as LA

def TimeStepSys(Sys,signal,xInit):
    """
    Time-step system x(k+1) = Ax(k) + Bu(k).
    
    Parameters
    ----------
    Sys : dictionary representing a LTI system.
    signal : 2-d array for control input.
    xInit : array for initial condition.

    Returns
    -------
    XTraj : 2-d array representing state trajectory.

    """
    
    if 'A' in Sys:
        A = Sys['A']
        B = Sys['B']
    elif 'Ar' in Sys:
        A = Sys['Ar']
        B = Sys['Br']

    nSteps = signal.shape[1]
    XTraj = np.zeros((xInit.shape[0],nSteps + 1))
    XTraj[:,0:1] = xInit
    
    for k in range(nSteps):
        XTraj[:,k+1:k+2] = A @ XTraj[:,k:k+1] + B @ signal[:,k:k+1]
    
    return XTraj

def genAnorm(Sys,nTimes):
    """
    Compute exact value of \|A^k\|_2 (AtrueNorm) for intrusive approach.

    Parameters
    ----------
    Sys : dictionary representing a LTI system.
    nTimes : length of control input.

    Returns
    -------
    AtrueNorm : array representing the norm of \|A^k\|_2 for values of k.

    """
    
    A = Sys['A']
    AtrueNorm = np.zeros((1,nTimes))
    
    # compute true norm of powers of A
    
    tempA = A
    
    for k in range(nTimes):
        AtrueNorm[:,k:k+1] = LA.norm(tempA, ord = 2)
        tempA = A @ tempA
    
    return AtrueNorm

def genBasis(Sys,signal,xInit,rdim):    
    """
    Generate basis matrix via POD.
    
    Parameters
    ----------
    Sys : dictionary representing a LTI system.
    signal : 2-d array for control input.
    xInit : array for initial condition.
    rdim : reduced dimension.

    Returns
    -------
    V : basis matrix whose columns are basis vectors.

    """
    
    XTraj = TimeStepSys(Sys, signal, xInit)
    
    # compute svd
    _ , _ , Vh = LA.svd(XTraj.T)
    V = Vh.T
    V = V[:, :rdim]
    
    return V
    
def genRedSys(Sys,V):
    """
    Generate reduced model operators 'Ar', 'Br' in xr(k+1) = Ar xr(k) + Br u(k)
    via projection.

    Parameters
    ----------
    Sys : dictionary representing a LTI system.
    V : basis matrix whose columns are basis vectors.

    Returns
    -------
    redModel : dictionary with keys 'Ar', 'Br'.

    """
    
    A = Sys['A']
    B = Sys['B']
    
    redModel = dict()
    redModel['Ar'] = V.T @ (A @ V)
    redModel['Br'] = V.T @ B
    
    return redModel


def genErrOp(Sys,V):    
    """
    Generate error operators M1, M2, M3, M4 defined in paper.

    Parameters
    ----------
    Sys : dictionary representing a LTI system.
    V : basis matrix whose columns are basis vectors.

    Returns
    -------
    ErrOp : dictionary with keys 'M1', 'M2', 'M3', 'M4'.

    """
    
    A = Sys['A']
    B = Sys['B']
    
    ErrOp = dict()
    ErrOp['M1'] = V.T @ (A.T @ A) @ V
    ErrOp['M2'] = B.T @ B
    ErrOp['M3'] = B.T @ A @ V
    ErrOp['M4'] = V.T @ V
    
    return ErrOp


    
def computeResidNormTraj(redModel,ErrOp,XrTraj,signal):
    """
    Compute residual norm from operators. Residual norm defined in paper in 
    terms of M1, M2, M3, M4, Ar, Br.

    Parameters
    ----------
    redModel : dictionary representing reduced system.
    ErrOp : dictionary containing error operators.
    XrTraj : 2-d array for trajectory of Xr (reduced state).
    signal : 2-d array for control input.

    Returns
    -------
    residNormTraj : 2-d array for trajectory of residual norm computed using 
                    M1, M2, M3, M4, Ar, Br.

    """
    
    M1 = ErrOp['M1']
    M2 = ErrOp['M2']
    M3 = ErrOp['M3']
    M4 = ErrOp['M4']
    
    Ar = redModel['Ar']
    Br = redModel['Br']
    
    nSteps = XrTraj.shape[1] - 1
    residNormTraj = np.zeros((1,nSteps))
    
    for k in range(nSteps):
        xCurr = XrTraj[:,k:k+1]
        xNext = XrTraj[:,k+1:k+2]
        u = signal[:,k:k+1]
        residNormSqTraj = xCurr.T @ M1 @ xCurr + u.T @ M2 @ u + \
                            2 * u.T @ M3 @ xCurr + xNext.T @ M4 @ xNext - \
                                2 * xNext.T @ Ar @ xCurr - 2 * xNext.T @ Br @ u
        
        residNormTraj[:,k:k+1] =  np.sqrt(residNormSqTraj)
    
    return residNormTraj