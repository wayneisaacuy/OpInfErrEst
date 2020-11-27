#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wayne Isaac Tan Uy, PhD
22 Nov 2020
"""

import numpy as np
import numpy.linalg as LA
from IntrusiveROM import *
from OpInf import *

def RunOpInf(Sys,TimeStepBlackBoxSys,signal,xInit,rdim,nSkip,Anorm,gamma):
    """
    Run operator inference to generate reduced model and error estimate from data.

    Parameters
    ----------
    Sys : dictionary representing a LTI system.
    TimeStepBlackBoxSys: a function to query the blackbox system for 
                         non-intrusive model reduction.
    signal : dictionary with keys 'Ubasis', 'Utrain', 'Utest' 
             representing the control inputs. 
    xInit : dictionary with keys 'x0basis', 'x0train', 'x0test' representing 
            the initial condition.
    rdim : reduced dimension.
    nSkip : number of states to skip in re-projection algorithm.
    Anorm : dictionary representing exact value or bounds for \|A^k\|_2.
    gamma : Tparameter in estimating upper bound for \|A^k\|_2.

    Returns
    -------
    errMat : array of length 3 measuring error of reduced model, intrusive 
             error estimate and non-intrusive error estimate.
    IntROM : dictionary representing the intrusive reduced system.
    IntErrOp: dictionary representing the intrusive error operators.
    NonIntROM: dictionary representing the non-intrusive reduced system.
    NonIntErrOp: dictionary representing the non-intrusive error operators.
     
    """
    
    # unpack system
    
    Ubasis = signal['Ubasis']
    Utrain = signal['Utrain']
    Utest = signal['Utest']
    x0basis = xInit['x0basis']
    x0train = xInit['x0train']
    x0test = xInit['x0test']
    AtrueNorm = Anorm['AtrueNorm']
    AnormBnd = Anorm['AnormBnd']
    AtrueNorm = np.flip(AtrueNorm, axis = 1)
    AnormBnd = (gamma ** 0.5) * np.flip(AnormBnd, axis = 1)

    # training phase
    
    V = genBasisNonInt(TimeStepBlackBoxSys,Ubasis,x0basis,rdim)
    
    # intrusive approach
    
    IntROM = genRedSys(Sys,V)
    IntErrOp = genErrOp(Sys,V) 
    
    # non-intrusive approach
    
    ReProjData = ReProj(TimeStepBlackBoxSys,Utrain,x0train,V,nSkip)
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
    NonIntErrBndTmp = AnormBnd * ResidNormTrajNonInt
    NonIntErrBndTmp = np.hstack((np.array([[0]]), NonIntErrBndTmp))
    NonIntErrBnd = np.cumsum(NonIntErrBndTmp)[None,:]
    
    # save error quantities

    IntErrRelAve = np.sum(XrIntErr)/(nTestSteps * np.sum(XTrajTestNorm))
    IntErrBndRelAve =  np.sum(IntErrBnd)/(nTestSteps * np.sum(XTrajTestNorm))
    NonIntErrBndRelAve = np.sum(NonIntErrBnd)/(nTestSteps * np.sum(XTrajTestNorm))
    IntResidNormAve = np.sum(ResidNormTrajInt)/nTestSteps
    NonIntResidNormAve = np.sum(ResidNormTrajNonInt)/nTestSteps
    
    errMat = np.array([IntErrRelAve, IntErrBndRelAve, NonIntErrBndRelAve, IntResidNormAve, NonIntResidNormAve])
    
    return errMat, IntROM, IntErrOp, NonIntROM, NonIntErrOp