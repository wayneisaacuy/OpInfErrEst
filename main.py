#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wayne Isaac Tan Uy, PhD
22 Nov 2020
"""

import random
import numpy as np
from HeatFOMBwdEul import *
from IntrusiveROM import *
from OpInf import *
from RunOpInf import *
from genBlackBoxSys import *
import matplotlib.pyplot as plt

# set seed

random.seed(1)

# specify parameters of heat system

N = 133 # state dimension
mu = 0.1 # parameter
deltaT = 0.01 # time discretization

# generate full system for intrusive model reduction

FOM = genHeatFOM(N,mu,deltaT)

# generate blackbox system for non-intrusive model reduction

TimeStepBlackBoxSys = genBlackBox(FOM)

# specify signals

Tend = 5 # end time
timeVals = np.arange(0, Tend + deltaT, deltaT)
tsteps = timeVals.shape[0]
Ubasis = np.exp(timeVals) * np.sin(20 * np.pi * np.arange(tsteps) / tsteps)
Ubasis = Ubasis.reshape(-1,Ubasis.shape[0])
Ubasis = Ubasis[:,:-1]

Utrain = np.random.normal(0, 1, size=(1,tsteps-1))
Utrain[0,0] = 0

Utest = np.exp(timeVals) * np.sin(12 * np.pi * np.arange(tsteps) / tsteps)
Utest = Utest.reshape(-1,Utest.shape[0])
Utest = Utest[:,:-1]

signal = dict()
signal['Ubasis'] = Ubasis
signal['Utrain'] = Utrain
signal['Utest'] = Utest

# specify initial conditions

xInit = dict()
xInit['x0basis'] = np.zeros((N,1))
xInit['x0train'] = np.zeros((N,1))
xInit['x0test'] = np.zeros((N,1))

# specify parameters for operator inference

nSkip = 5 # skip states in re-projection algorithm
M = 25 # parameter in probabilistic error estimation
gamma = 1 # parameter in probabilistic error estimation
rdimList = list(range(1,9)) # list of reduced dimensions

# pre-compute bound and exact value for norm of powers of A

AtrueNorm = genAnorm(FOM,tsteps-1)
AnormBnd = genAnormBnd(TimeStepBlackBoxSys,Ubasis.shape[0],N,tsteps-1,M)
#AnormBnd = genAnormBnd(FOM,tsteps-1,M)

Anorm = dict()
Anorm['AtrueNorm'] = AtrueNorm
Anorm['AnormBnd'] = AnormBnd

# run operator inference

errMat = []
IntROM = []
IntErrOp = []
NonIntROM = []
NonIntErrOp = []

for k in range(len(rdimList)):
    
    errMatrdim, IntROMdim, IntErrOpdim, NonIntROMdim, NonIntErrOpdim \
        = RunOpInf(FOM,TimeStepBlackBoxSys,signal,xInit,rdimList[k],nSkip,\
                   Anorm,gamma)
    
    errMat.append(errMatrdim)
    IntROM.append(IntROMdim)
    IntErrOp.append(IntErrOpdim)
    NonIntROM.append(NonIntROMdim)
    NonIntErrOp.append(NonIntErrOpdim)

errMat = np.array(errMat)
errMat = errMat.T
    
# plot results

# relative average error

fig, ax = plt.subplots()
ax.plot(rdimList, errMat[0,:], marker='o', label = 'Operator inference error')
ax.plot(rdimList, errMat[1,:], marker='s', label = 'Intrusive error estimate')
ax.plot(rdimList, errMat[2,:], marker ='v', label = 'Learned error estimate')

ax.set_yscale("log")
ax.legend()
ax.set(xlabel='reduced dimension', ylabel='rel ave state err over time',
       title='Heat equation example')
ax.grid()

plt.show()

# residual norm average

fig, ax = plt.subplots()
ax.plot(rdimList, errMat[3,:], marker='s', label = 'intrusive model reduction')
ax.plot(rdimList, errMat[4,:], marker='v', label = 'operator inference')

ax.set_yscale("log")
ax.legend()
ax.set(xlabel='reduced dimension', ylabel='ave residual norm over time',
       title='Heat equation example')
ax.grid()

plt.show()