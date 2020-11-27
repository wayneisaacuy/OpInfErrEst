#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wayne Isaac Tan Uy, PhD
27 Nov 2020
"""

import numpy as np

def genBlackBox(Sys):
    """
    Generate black box system for non-intrusive model reduction.
    
    Parameters
    ----------
    Sys : dictionary representing a LTI system.

    Returns
    -------
    TimeStepBlackBoxSy : a function that time-steps the blackbox system if the 
                         control input and initial condition are specified.

    """
    A = Sys['A']
    B = Sys['B']
    
    def TimeStepBlackBoxSys(signal,xInit):
        """
        Time step the blackbox system.
        
        Parameters
        ----------
        signal : 2-d array for control input.
        xInit : array for initial condition.

        Returns
        -------
        XTraj : 2-d array representing state trajectory.
        
        """
        
        nSteps = signal.shape[1]
        XTraj = np.zeros((xInit.shape[0],nSteps + 1))
        XTraj[:,0:1] = xInit
    
        for k in range(nSteps):
            XTraj[:,k+1:k+2] = A @ XTraj[:,k:k+1] + B @ signal[:,k:k+1]
    
        return XTraj
    
    return TimeStepBlackBoxSys