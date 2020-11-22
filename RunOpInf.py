#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wayne Isaac Tan Uy, PhD
22 Nov 2020
"""

import numpy as np
import numpy.linalg as LA
from ROMhelper import *
from OpInf import *

def RunOpInf(Sys,signal,xInit,rdim,nSkip,AnormBnd,gamma):
    """
    Run operator inference to generate reduced model and error estimate from data.

    Parameters
    ----------
    Sys : dictionary representing a LTI system.
    signal : dictionary with keys 'Ubasis', 'Utrain', 'Utest' 
            representing the control inputs. 
    xInit : dictionary with keys 'x0basis', 'x0train', 'x0test' representing 
            the initial condition.
    rdim : reduced dimension.
    nSkip : number of states to skip in re-projection algorithm.
    AnormBnd : dictionary representing bounds for \|A^k\|_2.
    gamma : Tparameter in estimating upper bound for \|A^k\|_2.

    Returns
    -------
    errMat : array of length 3 measuring error of reduced model, intrusive 
             error estimate and non-intrusive error estimate.

    """
    
    # unpack system
    
    Ubasis = signal['Ubasis']
    Utrain = signal['Utrain']
    Utest = signal['Utest']
    x0basis = xInit['x0basis']
    x0train = xInit['x0train']
    x0test = xInit['x0test']
    AtrueNorm = AnormBnd['AtrueNorm']
    XiSample = AnormBnd['XiSample']
    AtrueNorm = np.flip(AtrueNorm, axis = 1)
    XiSample = (gamma ** 0.5) * np.flip(XiSample, axis = 1)

    # training phase
    
    V = genBasis(Sys,Ubasis,x0basis,rdim)
    
    # intrusive approach
    
    IntROM = genRedSys(Sys,V)
    IntErrOp = genErrOp(Sys,V) 
    
    # non-intrusive approach
    
    ReProjData = ReProj(Sys,Utrain,x0train,V,nSkip)
    NonIntROM = genNonIntROM(ReProjData)
    NonIntErrOp = genNonIntEE(ReProjData,NonIntROM,V)
    
    # testing phase
    
    XTrajTest = TimeStepSys(Sys,Utest,x0test)
    XTrajTestNorm = LA.norm(XTrajTest, axis = 0)[None,:]
    nTestSteps = XTrajTestNorm.shape[1]
    
    # intrusive approach
    
    XrIntTrajTest = TimeStepSys(IntROM,Utest,V.T @ x0test)
    XrIntErr = LA.norm(XTrajTest - V @ XrIntTrajTest, axis = 0)
    ResidNormTrajInt = computeResidNormTraj(IntROM,IntErrOp,XrIntTrajTest,Utest)
    IntErrBndTmp = AtrueNorm * ResidNormTrajInt
    IntErrBndTmp = np.hstack((np.array([[0]]), IntErrBndTmp))
    IntErrBnd = np.cumsum(IntErrBndTmp)[None,:]
    
    # non-intrusive approach

    XrNonIntTrajTest = TimeStepSys(NonIntROM,Utest,V.T @ x0test)
    XrNonIntErr = LA.norm(XTrajTest - V @ XrNonIntTrajTest, axis = 0)
    ResidNormTrajNonInt = computeResidNormTraj(NonIntROM,NonIntErrOp,XrNonIntTrajTest,Utest)
    NonIntErrBndTmp = XiSample * ResidNormTrajNonInt
    NonIntErrBndTmp = np.hstack((np.array([[0]]), NonIntErrBndTmp))
    NonIntErrBnd = np.cumsum(NonIntErrBndTmp)[None,:]
    
    # save error quantities

    IntErrRelAve = np.sum(XrIntErr)/(nTestSteps * np.sum(XTrajTestNorm))
    IntErrBndRelAve =  np.sum(IntErrBnd)/(nTestSteps * np.sum(XTrajTestNorm))
    NonIntErrBndRelAve = np.sum(NonIntErrBnd)/(nTestSteps * np.sum(XTrajTestNorm))
    
    errMat = np.array([IntErrRelAve, IntErrBndRelAve, NonIntErrBndRelAve])
    
    return errMat