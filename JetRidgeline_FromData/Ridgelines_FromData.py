#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ridgelines_FromData.py
Read pre-existing ridgeline data and load into internal data structures required 
for creating edge points and sections.
Created by LizWhitehead - 21/06/2025
"""

import JetModelling_MapSetup as JMS
import JetModelling_MapAnalysis as JMA
import JetRidgeline_FromData.RLFDSetup as RLFDS
import JetRidgeline_FromData.RidgeToolkit_FromData as RLFDTK
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
    flux_array = RLFDS.Setup()

    # Create a cutout array, to reduce processing time
    flux_array = CreateCutout(flux_array)

    # Load existing ridgeline data
    ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2 = RLFDTK.LoadRidgelineData(flux_array)

    print('Time taken for ridgelines to draw = ' + str((time.time()-start_time)/60),'m')
    resetwarnings()

    return flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2

#############################################

def CreateCutout(flux_array):

    """
    Creates a cutout and update any necessary coordinates to be relative to the cutout.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    Returns
    -----------
    cutout_flux_array - cutout image array

    """

    # Create the cutout array
    cutout_offset = JMA.GetCutoutOffset(JMS.mCentre, np.array([JMS.sSize[1],JMS.sSize[0]]))
    x_start = cutout_offset[0]; x_end = floor(x_start + JMS.sSize[0])
    y_start = cutout_offset[1]; y_end = floor(y_start + JMS.sSize[1])
    cutout_flux_array = np.copy(flux_array[y_start:y_end, x_start:x_end])

    return cutout_flux_array