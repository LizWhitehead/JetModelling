#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetParameters.py
Compute jet parameters for both arms of the jet
Created by LizWhitehead - 08/04/2025
"""

import time
import JetParameters.JPSetup as JPS
from JetParameters.JetParameterToolkit import GetJetParameters
from warnings import resetwarnings

def ComputeJetParameters(map_type, source_name, section_parameters1, section_parameters2):

    """
    Compute jet parameters for each arm of the jet.

    Parameters
    -----------
    map_type - string

    source_name - string

    section_parameters1 - 2D array, shape(n,12)
                          Array with section points (x,y * 4), distance from source
                          and computed parameters for one arm of the jet

    section_parameters2 - 2D array, shape(n,12)
                          Array with section points (x,y * 4), distance from source
                          and computed parameters for other arm of the jet
    """
    
    # Set up the directory structure for jet parameters computation
    JPS.Setup(source_name, map_type)

    print('Starting jet parameters computation.')

    try:
        start_time = time.time()

        # Get parameter values along each arm of the jet
        GetJetParameters(section_parameters1, section_parameters2)

        print('Time taken to compute jet parameters = ' + str((time.time()-start_time)/(60*60)),'h')
    except:
        print('Error occurred computing jet parameters')

    resetwarnings()

