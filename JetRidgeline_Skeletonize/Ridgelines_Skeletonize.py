#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ridgelines_Skeletonize.py
Create a ridgeline via skimage skeletonize, as developed
by Hugh Dickenson, and load into internal data structures
required for creating edge points and sections.
Created by LizWhitehead - 24/08/2025
"""

import JetModelling_MapSetup as JMS
import JetModelling_MapAnalysis as JMA
import JetRidgeline_Skeletonize.RLSSetup as RLSS
import JetRidgeline_Skeletonize.RidgeToolkit_Skeletonize as RSTK
import numpy as np
from warnings import simplefilter, resetwarnings
from math import floor
import time

def CreateRidgelines():
    simplefilter(action="ignore")

    print('Starting ridgeline drawing process')
    print('-------------------------------------------')
    start_time = time.time()

    # Set up the directory structure for ridgeline processing. Produce 2D flattened cutout map.
    flux_array = RLSS.Setup()

    # Load existing ridgeline data
    ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2 = RSTK.CreateAndLoadSkeletonRidgeline(flux_array)

    # Create a cutout array, to reduce processing time
    flux_array, ridge1, ridge2 = CreateCutout(flux_array, ridge1, ridge2)

    print('Time taken for ridgelines to draw = ' + str((time.time()-start_time)/60),'m')
    resetwarnings()

    return flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2

#############################################

def CreateCutout(flux_array, ridge1, ridge2):

    """
    Creates a cutout and update any necessary coordinates to be relative to the cutout.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    ridge1 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for one side of the jet
    
    ridge2 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for other side of the jet

    Returns
    -----------
    cutout_flux_array - cutout image array

    cutout_ridge1 - Array of ridgepoint co-ordinates for the cutout
    
    cutout_ridge2 - Array of ridgepoint co-ordinates for the cutout

    """

    # Create the cutout array
    cutout_offset = JMA.GetCutoutOffset(JMS.mCentre, np.array([JMS.sSize[1],JMS.sSize[0]]))
    x_start = cutout_offset[0]; x_end = floor(x_start + JMS.sSize[0])
    y_start = cutout_offset[1]; y_end = floor(y_start + JMS.sSize[1])
    cutout_flux_array = np.copy(flux_array[y_start:y_end, x_start:x_end])

    # Update original co-ordinates to be relative to the cutout
    cutout_ridge1 = ridge1 - cutout_offset                          # Ridgepoints side 1
    cutout_ridge2 = ridge2 - cutout_offset                          # Ridgepoints side 2

    return cutout_flux_array, cutout_ridge1, cutout_ridge2