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
from JetRidgeline.RidgeToolkit import CreateCutouts, TrialSeries
from warnings import resetwarnings

def Draw_Ridgelines():
    print('Creating cutouts')
    start_time = time.time()
    CreateCutouts()
    print('Time taken to make cutout = ' + str((time.time()-start_time)/(60*60)),'h')

    print('Starting Ridgeline drawing process.')
    start_time = time.time()
    TrialSeries(RLC.R, RLC.dphi)
    print('Time taken for Ridgeline to draw = ' + str((time.time()-start_time)/(60*60)),'h')
    resetwarnings()

