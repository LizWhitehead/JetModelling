#!/usr/bin/env python 3
# coding: utf-8

# ## Ridgeline Code for DR2
# 
# This is a script version of notebook from BB containing the ridgeline code for DR2.  This is the preparation for running the likelihood ratio code and only needs to be run once for each dataset. 
# Modified by MJH to use tools from the sizeflux code

# Imports

import time
import JetRidgeline.RidgelineFiles as RLF
import JetRidgeline.RLConstants as RLC
import JetRidgeline.RLSetup as RLS
import JetRidgeline.RLGlobal as RLG
from JetRidgeline.RidgeToolkit import CreateCutouts, TrialSeries
from warnings import resetwarnings

def CreateRidgelinesAndSections(map_file, map_type):
    
    # Set up the directory structure for ridgeline processing. Produce 2D flattened map and thresholded npy cutout.
    RLS.Setup(map_file, map_type)

    print('Creating cutouts')
    start_time = time.time()
    CreateCutouts()
    print('Time taken to make cutout = ' + str((time.time()-start_time)/(60*60)),'h')

    print('Starting ridgeline/edgepoint drawing process.')
    start_time = time.time()
    section_parameters1, section_parameters2 = TrialSeries(RLC.R, RLC.dphi)
    print('Time taken for ridgelines and sections to draw = ' + str((time.time()-start_time)/(60*60)),'h')
    resetwarnings()

    return RLG.sName, section_parameters1, section_parameters2

