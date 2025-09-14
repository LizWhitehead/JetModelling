#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ridgelines_Skeletonize.py
Create a ridgeline via skimage skeletonize, as developed
by Hugh Dickenson, and load into internal data structures
required for creating edge points and sections.
Created by LizWhitehead - 24/08/2025
"""

import time
import JetRidgeline_Skeletonize.RLSSetup as RLSS
import JetRidgeline_Skeletonize.RidgeToolkit_Skeletonize as RSTK
from warnings import resetwarnings

def CreateRidgelines():

    print('Starting ridgeline drawing process')
    print('-------------------------------------------')
    start_time = time.time()

    # Set up the directory structure for ridgeline processing. Produce 2D flattened cutout map.
    flux_array = RLSS.Setup()

    # Load existing ridgeline data
    ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2 = RSTK.CreateAndLoadSkeletonRidgeline(flux_array)

    print('Time taken for ridgelines to draw = ' + str((time.time()-start_time)/60),'m')
    resetwarnings()

    return flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2

