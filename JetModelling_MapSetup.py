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
map_count = 3                   # count of map files in this run
sName = '3C31'                  # Source name

###########################################################################################################
# Set these parameters from the map FITS file
###########################################################################################################
map_files =  np.array(['C:/Maps/3C31.HGEOM2.FITS',                                                                  # FITS map file
                       'C:/Maps/3C31_VLA_360MHz_9arcs.fits',
                       'C:/Maps/3C31_GMRT_615MHz_5arcs.fits'])
sSizes    =  np.array([np.array([455,815]), np.array([510,900]), np.array([520,1000])])                             # Source size in pixels (x,y sides of containing rectangle, centred on mCentre)
rmSources =  np.array([np.array([[798,860,10],[952,1090,8],[894,732,15],[764,735,16],[761,780,12],[1038,1098,12]]), # Data to remove bright stars (centre x, centre y, +/- no. of pixels); array of -1's if none
                       np.array([[2075,2161,9],[1918,1847,20],[1922,1889,12],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]]),
                       np.array([[1538,1668,10],[1516,1884,15],[1402,1756,15],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]])])
bgRMSs_manual =  np.array([0.0002, 0.0003, 0.0004]); bgMeans_manual = np.array([-0.000079, -0.000062, -0.000081])   # Background RMS and mean flux values read manually from the map (empty arrays if not present)

###########################################################################################################
# Parameters used for from-data ridgelines.
###########################################################################################################
ridgelines_from_data_side1s = np.array(['C:/JetModelling_FromData/3C31_1.6GHz_ridge_coords1.txt',                   # Input ridgeline data file for side1
                                       'C:/JetModelling_FromData/3C31_360MHz_ridge_coords1.txt',
                                       'C:/JetModelling_FromData/3C31_615MHz_ridge_coords1.txt'])
ridgelines_from_data_side2s = np.array(['C:/JetModelling_FromData/3C31_1.6GHz_ridge_coords2.txt',                   # Input ridgeline data file for side2
                                       'C:/JetModelling_FromData/3C31_360MHz_ridge_coords2.txt',
                                       'C:/JetModelling_FromData/3C31_615MHz_ridge_coords2.txt'])

###########################################################################################################
# Parameters used for map plotting.
###########################################################################################################
ImFractions  = np.array([1.0,1.0,1.0])                                                                              # The fraction of the source the final image is cut down to for plotting
vmins        = np.array([0,0,0])                                                                                    # Linear mapping colour range minimum
vmaxs        = np.array([0.010,0.010,0.010])                                                                        # Linear mapping colour range maximum

###########################################################################################################
# Parameters used for skeletonize ridgelines (these parameters are highly sensitive to the map).
# Modify these parameters to achieve the best ridgeline. Run multiple times with the ridgeline_only flag
# set to True until the best ridgeline is achieved.
###########################################################################################################
CutdownSize0s =            np.array([1000,1270,1000])                                                               # Cut-down size of image (y axis) for skeletonize
CutDownSize1s =            np.array([1000,1270,1000])                                                               # Cut-down size of image (x axis) for skeletonize
GaussSigmaFilters =        np.array([35,40,35])                                                                     # Sigma level for thresholding
ContoursLevelPercs =       np.array([80,80,80])                                                                     # Percentile level for finding contours
MaxRemoveSmallHolesAreas = np.array([200,1000,1000])                                                                # Maximum area of "holes" in skeleton that will be removed
MaximumLoopJumpPixelss =   np.array([20,20,20])                                                                     # Maximum jump between points, used in finding "loops"
nSig_ss =                  np.array([12,20,20])                                                                     # The multiple of sigma for RMS reduction; nan if not required
SplitInnerOuterSkeletons = np.array([True,True,True])                                                               # Flag for joining together skeletons for inner and outer jets
nSig_s_outers =            np.array([nan,nan,nan])                                                                  # Value used if joining together skeletons for inner and outer jets
JoinInterpolatePointss =   np.array([6,30,6])                                                                       # Number of points in the smoothing interpolation at the join

###########################################################################################################
# Parameters populated automatically from the FITS file header
###########################################################################################################
rdels = np.empty(map_count); ddels = np.empty(map_count)
beamsizes =  np.empty(map_count); beamareas = np.empty(map_count)
sRAs = np.empty(map_count); sDecs = np.empty(map_count)
bgRMSs = np.empty(map_count); bgMeans = np.empty(map_count)
freqs = np.empty(map_count)
mCentres = np.empty((map_count,2))

#############################################

def InitialiseMap(current_map_number):

    """
    Initialises all map-specific parameter values for the current map number

    Parameters
    -----------
    current_map_number - integer, zero-based
    """
    global map_number

    global map_file                                 # FITS map file
    global rdel                                     # Equivalent pixel value for RA in FITS file in degrees
    global ddel                                     # Equivalent pixel value for Dec in FITS file in degrees
    global beamsize                                 # Beam size (width along semi-major axis) in pixels
    global beamarea                                 # Beam area in pixels squared
    global freq                                     # Observation frequency
    global sRA                                      # Map centre RA in degrees
    global sDec                                     # Map centre Dec in degrees
    global mCentre                                  # Map centre (x,y) in pixels
    global sSize                                    # Source size in pixels (x,y sides of containing rectangle, centred on mCentre)
    global bgRMS                                    # Background flux RMS value in Jy/beam
    global bgMean                                   # Background flux mean pixel value in Jy/beam
    global rmSource                                 # Data to remove bright stars (centre x, centre y, +/- no. of pixels)

    global equiv_R_factor                           # Factor which determines equivalent distances on multiple maps
    global max_beamsize                             # Maximum beamsize for multiple jets

    global CutdownSize0                             # Cut-down size of image (y axis) for skeletonize
    global CutDownSize1                             # Cut-down size of image (x axis) for skeletonize
    global GaussSigmaFilter                         # Sigma level for thresholding
    global ContoursLevelPerc                        # Percentile level for finding contours
    global MaxRemoveSmallHolesArea                  # Maximum area of "holes" in skeleton that will be removed
    global MaximumLoopJumpPixels                    # Maximum jump between points, used in finding "loops"
    global nSig_s                                   # The multiple of sigma for RMS reduction; nan if not required
    global SplitInnerOuterSkeleton                  # Flag for joining together skeletons for inner and outer jets
    global nSig_s_outer                             # Value used if joining together skeletons for inner and outer jets
    global JoinInterpolatePoints                    # Number of points in the smoothing interpolation at the join

    global ridgelines_from_data_side1               # Input ridgeline data file for side1
    global ridgelines_from_data_side2               # Input ridgeline data file for side2

    global ImFraction                               # The fraction of the source the final image is cut down to for plotting
    global vmin                                     # Linear mapping colour range minimum
    global vmax                                     # Linear mapping colour range maximum

    # Set the map number
    map_number = current_map_number

    # Initialise all map-specific parameter values for this map number
    map_file = map_files[map_number]

    rdel = rdels[map_number]; ddel = ddels[map_number]
    sRA = sRAs[map_number]; sDec = sDecs[map_number]
    mCentre = mCentres[map_number]
    freq = freqs[map_number]
    beamsize = beamsizes[map_number]; beamarea = beamareas[map_number]
    bgRMS = bgRMSs[map_number]; bgMean = bgMeans[map_number]
    sSize = sSizes[map_number]
    rmSource = rmSources[map_number]
    equiv_R_factor = ddels[0] / ddels[map_number]
    max_beamsize = np.max(beamsizes)

    ridgelines_from_data_side1 = ridgelines_from_data_side1s[map_number]
    ridgelines_from_data_side2 = ridgelines_from_data_side2s[map_number]

    ImFraction = ImFractions[map_number]; vmin = vmins[map_number]; vmax = vmaxs[map_number]

    CutdownSize0 = CutdownSize0s[map_number]; CutDownSize1 = CutDownSize1s[map_number]
    GaussSigmaFilter = GaussSigmaFilters[map_number]
    ContoursLevelPerc = ContoursLevelPercs[map_number]
    MaxRemoveSmallHolesArea = MaxRemoveSmallHolesAreas[map_number]
    MaximumLoopJumpPixels = MaximumLoopJumpPixelss[map_number]
    nSig_s = nSig_ss[map_number]
    SplitInnerOuterSkeleton = SplitInnerOuterSkeletons[map_number]; nSig_s_outer = nSig_s_outers[map_number]; JoinInterpolatePoints = JoinInterpolatePointss[map_number]

    print(' ')
    print('--------------------------------------------------------------------------------------')
    print('Starting modelling for ' + sName + ' - map ' + str(map_number+1) + ' (' + map_file + ')')
    print('--------------------------------------------------------------------------------------')
