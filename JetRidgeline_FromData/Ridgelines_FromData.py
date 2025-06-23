#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ridgelines_FromData.py
Read pre-existing ridgeline data and load into internal data structures required 
for creating edge points and sections.
Created by LizWhitehead - 21/06/2025
"""

import time
import JetRidgeline_FromData.RLFDSetup as RLFDS
import JetRidgeline_FromData.RidgeToolkit_FromData as RLFDTK
import JetRidgeline_FromData.RLFDConstants as RLFDC
from warnings import resetwarnings

def CreateRidgelines():

    print('Starting ridgeline drawing process')
    print('-------------------------------------------')
    start_time = time.time()

    # Set up the directory structure for ridgeline processing. Produce 2D flattened cutout map.
    flux_array = RLFDS.Setup()

    # Load existing ridgeline data
    ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2 = RLFDTK.LoadRidgelineData(flux_array, RLFDC.R)

    print('Time taken for ridgelines to draw = ' + str((time.time()-start_time)/60),'m')
    resetwarnings()

    return flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2

