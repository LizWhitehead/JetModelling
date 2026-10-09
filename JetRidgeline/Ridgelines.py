#!/usr/bin/env python 3
# coding: utf-8
"""
Ridgelines_FromData.py
Create ridgeline data structures, required for creating edge points and sections.
Modified by:    LizWhitehead 13/11/2024
"""

import time
import JetModelling_MapSetup as JMS
import JetModelling_Constants as JMC
import JetRidgeline.RLSetup as RLS
import JetRidgeline.RidgeToolkit as RTK
import numpy as np

def CreateRidgelines(flux_array):

    print(' ')
    print('Starting ridgeline drawing process')
    print('-------------------------------------------')
    start_time = time.time()
    
    # Set up the directory structure for ridgeline processing. Produce 2D flattened map and thresholded npy cutout.
    RLS.Setup()

    # Copy the array, as it will be convolved during ridgeline processing
    ridge_flux_array = flux_array.copy()

    # Create ridgelines
    ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2 = RTK.TrialSeries(ridge_flux_array)

    print('Time taken for ridgelines to draw = ' + str((time.time()-start_time)/60),'m')

    return ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2
