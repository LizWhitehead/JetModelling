#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetParameters.py
Compute jet parameters for both sides of the jet
Created by LizWhitehead - 08/04/2025
"""

import time
import JetParameters.JPSetup as JPS
from JetParameters.JetParameterToolkit import GetJetParameters

def ComputeJetParameters(section_parameters1, section_parameters2):

    """
    Compute jet parameters for each side of the jet.

    Parameters
    -----------
    section_parameters1 - 2D array, shape(n,12)
                          Array with section points (x,y * 4), distance from source,
                          flux, volume and area for one side of the jet

    section_parameters2 - 2D array, shape(n,12)
                          Array with section points (x,y * 4), distance from source,
                          flux, volume and area for other side of the jet
    """

    print('Starting jet parameters computation')
    print('-------------------------------------------')
    start_time = time.time()
    
    # Set up the directory structure for jet parameters computation
    JPS.Setup()

    # Get parameter values along each side of the jet
    GetJetParameters(section_parameters1, section_parameters2)

    print('Time taken to compute jet parameters = ' + str((time.time()-start_time)/60),'m')
