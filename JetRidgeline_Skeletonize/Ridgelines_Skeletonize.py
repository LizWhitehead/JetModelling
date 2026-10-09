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
import time

def CreateRidgelines():

    print(' ')
    print('Starting ridgeline drawing process')
    print('-------------------------------------------')
    start_time = time.time()

    # Set up the directory structure for ridgeline processing.
    RLSS.Setup()

    # Load existing ridgeline data
    ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2 = RSTK.CreateAndLoadSkeletonRidgeline(flux_array)

    # Create a cutout array, to reduce processing time
    flux_array = JMA.GetCutout(JMS.map_file, JMS.mCentre, JMS.sSize)

    # Update original co-ordinates to be relative to the cutout
    cutout_offset = JMA.GetCutoutOffset(JMS.mCentre, np.array([JMS.sSize[1],JMS.sSize[0]]))
    cutout_ridge1 = ridge1 - cutout_offset                          # Ridgepoints side 1
    cutout_ridge2 = ridge2 - cutout_offset                          # Ridgepoints side 2

    print('Time taken for ridgelines to draw = ' + str((time.time()-start_time)/60),'m')

    return flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2
