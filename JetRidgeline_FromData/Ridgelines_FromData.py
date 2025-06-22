#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ridgelines_FromData.py
Read pre-existing ridgeline data and load into internal data structures required 
for creating edge points and sections.
Created by LizWhitehead - 21/06/2025
"""

import time
import JetModelling_MapSetup as JMS
import JetRidgeline_FromData.RLFDSetup as RFDS
import JetRidgeline_FromData.RidgeToolkit_FromData as RFDTK
from warnings import resetwarnings

def CreateRidgelines():

    print('Starting ridgeline drawing process')
    print('-------------------------------------------')
    start_time = time.time()

    # Set up the directory structure for ridgeline processing. Produce 2D flattened cutout map.
    flux_array = RFDS.Setup()

    # Load existing ridgeline data
    ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2 = RFDTK.LoadRidgelineData(flux_array)

    print('Time taken for ridgelines to draw = ' + str((time.time()-start_time)/60),'m')
    resetwarnings()

    return flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2

