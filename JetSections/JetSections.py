#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetSections.py
Divide both arms of the jet into sections.
Created by LizWhitehead - 30/04/2025
"""

import JetSections.JSSetup as JSS
from JetSections.JetSectionToolkit import GetEdgepointsAndSections
from warnings import resetwarnings
import time

def CreateJetSections(flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2):

    """
    Divide both arms of the jet into sections by finding edge points.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    ridge1 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for one arm of the jet
    
    ridge2 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for other arm of the jet

    phi_val1 - 1D array of ridgeline angles for each ridgepoint on
               one arm of the jet
    
    phi_val2 - 1D array of ridgeline angles for each ridgepoint on
               other arm of the jet

    Rlen1 - 1D array of disance from source for each ridgepoint on
            one arm of the jet
    
    Rlen2 - 1D array of disance from source for each ridgepoint on
            other arm of the jet
    
    Constants
    ---------

    Returns
    -----------
    section_parameters1 - 2D array, shape(n,12)
                          Array with section points (x,y * 4), distance from source,
                          flux and volume for one arm of the jet

    section_parameters2 - 2D array, shape(n,12)
                          Array with section points (x,y * 4), distance from source
                          flux and volume for other arm of the jet

    Notes
    -----------
    """
    
    # Set up the directory structure for jet sections processing
    JSS.Setup()

    print('Starting jet sections processing')
    print('-------------------------------------------')
    start_time = time.time()

    section_parameters1, section_parameters2 = GetEdgepointsAndSections(flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2)

    print('Time taken for jet sections processing = ' + str((time.time()-start_time)/60),'m')
    resetwarnings()

    return section_parameters1, section_parameters2
