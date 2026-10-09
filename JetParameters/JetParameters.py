#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetParameters.py
Compute jet parameters for both sides of the jet
Created by LizWhitehead - 08/04/2025
"""

import time
import JetParameters.JPSetup as JPS
from JetParameters.JetParameterToolkit import GetJetParameters, GetSpectralIndexParameters

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

    print(' ')
    print('Starting jet parameters computation')
    print('-------------------------------------------')
    start_time = time.time()
    
    # Set up the directory structure for jet parameters computation
    JPS.Setup()

    # Get parameter values along each side of the jet
    GetJetParameters(section_parameters1, section_parameters2)

    print('Time taken to compute jet parameters = ' + str((time.time()-start_time)/60),'m')

#############################################

def ComputeSpectralIndex(start_time, common_section_params1, common_section_params2):

    """
    Compute spectral index along each side of the jet.

    Parameters
    -----------
    start_time - start time for spectral index processing

    common_section_params1 - 3D array, shape(m,n,11)
                             Array with section points (x,y * 4), distance from source,
                             flux and area for one side of the jet.
                             n: number of sections
                             m: number of maps

    common_section_params2 - 3D array, shape(m,n,11)
                             Array with section points (x,y * 4), distance from source
                             flux and area for other side of the jet.
                             n: number of sections
                             m: number of maps
    """

    # Set up the directory structure for spectral index processing
    JPS.Setup_spectral_index()

    # Determine the spectral index along the jet
    GetSpectralIndexParameters(common_section_params1, common_section_params2)
    
    print('Time taken for spectral index processing = ' + str((time.time()-start_time)/60),'m')
