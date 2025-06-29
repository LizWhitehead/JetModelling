#!/usr/bin/env python 3
# coding: utf-8

# ## Ridgeline Code for DR2
# 
# This is a script version of notebook from BB containing the ridgeline code for DR2.  This is the preparation for running the likelihood ratio code and only needs to be run once for each dataset. 
# Modified by MJH to use tools from the sizeflux code

# Imports

import time
import JetModelling_MapSetup as JMS
import JetRidgeline.RLSetup as RLS
import JetRidgeline.RidgeToolkit as RTK
from warnings import resetwarnings

def CreateRidgelines():

    print('Starting ridgeline drawing process')
    print('-------------------------------------------')
    start_time = time.time()
    
    # Set up the directory structure for ridgeline processing. Produce 2D flattened map and thresholded npy cutout.
    RLS.Setup()

    print('Creating cutouts')
    start_time = time.time()
    RTK.CreateCutouts()
    print('Time taken to make cutout = ' + str((time.time()-start_time)/60),'m')

    # Create ridgelines
    ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2 = RTK.TrialSeries()

    flux_array = RTK.GetCutoutArray(JMS.sName)          # Return raw unconvolved data

    print('Time taken for ridgelines to draw = ' + str((time.time()-start_time)/60),'m')
    resetwarnings()

    return flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2

