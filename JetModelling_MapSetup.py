#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetModelling_MapSetup.py
Map-specific parameters for all maps in this run (same AGN, different frequencies)
Created by LizWhitehead - 18/04/2025
"""

import numpy as np
from math import nan

# Parameters for all maps in this run
map_count = 1                   # count of map files in this run
sName = 'NGC1044'               # Source name

###########################################################################################################
# Set these parameters from the map FITS file
###########################################################################################################
map_files =  np.array(['C:/Maps/img_do_not_touch_1728590175_sdp_l0_1024ch_NGC1044.k-cal_manualr0-MFS-I-image.fits'])               # FITS map file
rdels =      np.array([-0.000305555555555556 ])                                   # Equivalent pixel value for RA in FITS file in degrees
ddels =      np.array([0.000305555555555556 ])                                    # Equivalent pixel value for Dec in FITS file in degrees
beamsizes =  np.array([9.8])                                         # Beam size (width along semi-major axis) in pixels
beamareas =  np.array([73.774828994279])                                  # Beam area in pixels squared
freqs =      np.array([1283895507.8125])                                    # Observation frequency
sCentres =   np.array([np.array([1281,1281])])                          # Map source centre in pixels (x,y); array of nan's for a map if not present
# sCentres =   np.array([np.array([nan,nan])])                                # Map source centre in pixels (x,y); array of nan's for a map if not present
sRAs =       np.array([40.278375])                                          # Map centre RA in degrees
sDecs =      np.array([8.7338333333333])                                        # Map centre Dec in degrees
sSizes =     np.array([np.array([1200,2200])])                               # Source size in pixels (x,y sides of containing rectangle, centred on sCentre)
#bgRMSs =     np.array([0.000006])                                           # Background flux RMS value in Jy/beam
bgRMSs =     np.array([1e-05])                                           # Background flux RMS value in Jy/beam
bgMeans =    np.array([0.0])                                                # Background flux mean pixel value in Jy/beam
rmSources =    np.array([np.array([[1283,1534,12],[1289,1593,5],[1142,1579,20],[1201,795,10],[1235,1440,5],[1456,1475,25]])])        # Data to remove bright sources (centre x, centre y, +/- no. of pixels); array of -1's if none

###########################################################################################################
# Parameters used for from-data ridgelines.
###########################################################################################################
ridgelines_from_data_arm1s = np.array(['C:/JetModelling_FromData/NGC1044_Lucy_ridge1_new.txt'])   # Input data file for arm1
ridgelines_from_data_arm2s = np.array(['C:/JetModelling_FromData/NGC1044_Lucy_ridge2_new.txt'])   # Input data file for arm2

###########################################################################################################
# Parameters used for skeletonize ridgelines (these parameters are highly sensitive to the map).
# Modify these parameters to achieve the best ridgeline. Run multiple times with the ridgeline_only flag
# set to True until the best ridgeline is achieved.
###########################################################################################################
CutdownSize0s =            np.array([1000])                                 # Cut-down size of image (y axis) for skeletonize
CutDownSize1s =            np.array([1000])                                 # Cut-down size of image (x axis) for skeletonize
GaussSigmaFilters =        np.array([35])                                   # Sigma level for thresholding
ContoursLevelPercs =       np.array([80])                                   # Percentile level for finding contours
MaxRemoveSmallHolesAreas = np.array([200])                                 # Maximum area of "holes" in skeleton that will be removed
MaximumLoopJumpPixelss =   np.array([10])                                   # Maximum jump between points, used in finding "loops"
nSig_ss =                  np.array([12])                                   # The multiple of sigma for RMS reduction; nan if not required
SplitInnerOuterSkeletons = np.array([True])                                 # Flag for joining together skeletons for inner and outer jets
nSig_s_outers =            np.array([nan])                                  # Value used if joining together skeletons for inner and outer jets
JoinInterpolatePointss =   np.array([6])                                    # Value used if joining together skeletons for inner and outer jets

###########################################################################################################
# Parameters used for map plotting.
###########################################################################################################
ImFractions  = np.array([1.0])                                              # The fraction of the source the final image is cut down to for plotting
vmins        = np.array([21e-6])                                                # Linear mapping colour range minimum
vmaxs        = np.array([0.00175])                                            # Linear mapping colour range maximum

#############################################

def InitialiseMap(current_map_number):

    """
    Initialises all map-specific parameter values for the current map number

    Parameters
    -----------
    current_map_number - integer, zero-based
    """

    global map_number
    global map_file
    global rdel; global ddel; global equiv_R_factor
    global beamsize; global beamarea; global max_beamsize
    global freq
    global sCentre
    global sRA; global sDec
    global sSize
    global bgRMS; global bgMean
    global rmSource
    global CutdownSize0; global CutDownSize1
    global GaussSigmaFilter 
    global ContoursLevelPerc
    global MaxRemoveSmallHolesArea
    global MaximumLoopJumpPixels
    global nSig_s
    global SplitInnerOuterSkeleton; global nSig_s_outer; global JoinInterpolatePoints
    global ridgelines_from_data_arm1; global ridgelines_from_data_arm2
    global ImFraction; global vmin; global vmax

    # Set the map number
    map_number = current_map_number

    # Set the factor which determines equivalent distances along multiple jets
    equiv_R_factor = ddels[0] / ddels[map_number]

    # Set the maximum beamsize for multiple jets
    max_beamsize = np.max(beamsizes)

    # Initialise all map-specific parameter values for this map number
    map_file = map_files[map_number]
    rdel = rdels[map_number]; ddel = ddels[map_number]
    beamsize = beamsizes[map_number]; beamarea = beamareas[map_number]
    freq = freqs[map_number]
    sCentre = sCentres[map_number]
    sRA = sRAs[map_number]; sDec = sDecs[map_number]
    sSize = sSizes[map_number]
    bgRMS = bgRMSs[map_number]; bgMean = bgMeans[map_number]
    rmSource = rmSources[map_number]

    CutdownSize0 = CutdownSize0s[map_number]; CutDownSize1 = CutDownSize1s[map_number]
    GaussSigmaFilter = GaussSigmaFilters[map_number]
    ContoursLevelPerc = ContoursLevelPercs[map_number]
    MaxRemoveSmallHolesArea = MaxRemoveSmallHolesAreas[map_number]
    MaximumLoopJumpPixels = MaximumLoopJumpPixelss[map_number]
    nSig_s = nSig_ss[map_number]
    SplitInnerOuterSkeleton = SplitInnerOuterSkeletons[map_number]; nSig_s_outer = nSig_s_outers[map_number]; JoinInterpolatePoints = JoinInterpolatePointss[map_number]

    ridgelines_from_data_arm1 = ridgelines_from_data_arm1s[map_number]
    ridgelines_from_data_arm2 = ridgelines_from_data_arm2s[map_number]

    ImFraction = ImFractions[map_number]; vmin = vmins[map_number]; vmax = vmaxs[map_number]

    print(' ')
    print('--------------------------------------------------------------------------------------')
    print('Starting modelling for ' + sName + ' - map ' + str(map_number+1) + ' (' + map_file + ')')
    print('--------------------------------------------------------------------------------------')
    print(' ')
