#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wayne Isaac Tan Uy, PhD
22 Nov 2020
"""

import numpy as np
import numpy.linalg as LA

def genHeatFOM(N,mu,deltaT):
    """
    Constructs the discrete system 
    x(k+1) = Ax(k) + Bu(k) for the heat equation example in paper using
    finite element method and backward Euler. 
    
    Parameters
    ----------
    N : state dimension.
    mu : parameter value.
    deltaT : time step discretization.

    Returns
    -------
    heatFOM : dictionary with keys 'A', 'B'. 
    
    """
    
    xStart = 0
    xEnd = 1
    dx = (xEnd - xStart)/N
    
    dxVec = dx * np.ones(N)
    invdxVec = (1/dx) * np.ones(N)
    
    # assemble mass matrix
    M = np.diag(1/6 * dxVec[:-1], k = -1) + \
        np.diag(1/6 * dxVec[:-1], k = 1) + \
        np.diag(2/3 * dxVec, k = 0)
    M[-1,-1] = 0.5*M[-1,-1]
    
    # assemble stiffness matrix
    K = np.diag(-invdxVec[:-1], k = -1) + \
        np.diag(-invdxVec[:-1], k = 1) + \
        np.diag(2 * invdxVec, k = 0)    
    K = -mu*K
    K[-1,-1] = 0.5*K[-1,-1]
    
    # assemble rhs matrix
    rhs = np.zeros((N,1))
    rhs[-1] = mu
    
    # assemble continuous LTI matrices
    A = LA.solve(M,K)
    B = LA.solve(M,rhs)
    
    # assemble discrete LTI matrices
    Adisc = LA.solve(np.identity(N) - deltaT*A, np.identity(N))
    Bdisc = deltaT*LA.solve(np.identity(N) - deltaT*A, B)
    
    heatFOM = dict()
    heatFOM['A'] = Adisc
    heatFOM['B'] = Bdisc
    
    return heatFOM
