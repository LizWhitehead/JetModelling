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
map_files =  np.array(['C:/Maps/3C31.HGEOM2.FITS',
                       'C:/Maps/3C31_VLA_360MHz_9arcs.fits',
                       'C:/Maps/3C31_GMRT_615MHz_5arcs.fits'])              # FITS map file
rdels =      np.array([-0.0003611111024,
                       -0.0004166666666667,
                       -0.0002777777845])                                   # Equivalent pixel value for RA in FITS file in degrees
ddels =      np.array([0.0003611111024,
                       0.0004166666666667,
                       0.0002777777845])                                    # Equivalent pixel value for Dec in FITS file in degrees
beamsizes =  np.array([3.846184708166425,
                       5.7979838053387365,
                       5.414759868962811])                                  # Beam size (width along semi-major axis) in pixels
beamareas =  np.array([16.76195591180468,
                       20.79097537021711,
                       27.43927279185161])                                  # Beam area in pixels squared
freqs =      np.array([1.636e9,
                       3.599498640499e8,
                       6.15250000000e8])                                    # Observation frequency
# sCentres =   np.array([np.array([920.5,957.25]),
#                        np.array([2047.5,2046.25]),
#                        np.array([1497.75,1497.0])])                         # Map source centre in pixels (x,y); array of nan's for a map if not present
sCentres =   np.array([np.array([nan,nan]),
                       np.array([nan,nan]),
                       np.array([nan,nan])])                                # Map source centre in pixels (x,y); array of nan's for a map if not present
sRAs =       np.array([16.8217500000,
                       16.85395833273,
                       16.8537683301])                                      # Map centre RA in degrees
sDecs =      np.array([32.4255277778,
                       32.41255555384,
                       32.4125070566])                                      # Map centre Dec in degrees
sSizes =     np.array([np.array([500,500]),
                       np.array([500,500]),
                       np.array([500,1000])])                               # Source size in pixels (x,y sides of containing rectangle)
bgRMSs =     np.array([0.000198,
                       0.00026,
                       0.000313])                                           # Background flux RMS value in Jy/beam
bgMeans =    np.array([-0.000078874466,
                       0.00014237006,
                       -0.00016326427])                                     # Background flux mean pixel value in Jy/beam

###########################################################################################################
# Parameters used for skeletonize ridgelines (these parameters are highly sensitive to the map).
# Modify these parameters to achieve the best ridgeline. Run multiple times with the ridgeline_only flag
# set to True until the best ridgeline is achieved.
###########################################################################################################
CutdownSize0s =            np.array([1000,
                                     1270,
                                     1000])                                 # Cut-down size of image (y axis) for skeletonize
CutDownSize1s =            np.array([1000,
                                     1270,
                                     1000])                                 # Cut-down size of image (x axis) for skeletonize
GaussSigmaFilters =        np.array([35,
                                     40,
                                     35])                                   # Sigma level for thresholding
ContoursLevelPercs =       np.array([80,
                                     80,
                                     80])                                   # Percentile level for finding contours
MaxRemoveSmallHolesAreas = np.array([200,
                                     1000,
                                     1000])                                 # Maximum area of "holes" in skeleton that will be removed
MaximumLoopJumpPixelss =   np.array([20,
                                     20,
                                     20])                                   # Maximum jump between points, used in finding "loops"
nSig_ss =                  np.array([12,
                                     20,
                                     20])                                   # The multiple of sigma for RMS reduction; nan if not required
SplitInnerOuterSkeletons = np.array([True,
                                     True,
                                     True])                                 # Flag for joining together skeletons for inner and outer jets
nSig_s_outers =            np.array([nan,
                                     nan,
                                     nan])                                  # Value used if joining together skeletons for inner and outer jets
JoinInterpolatePointss =   np.array([6,
                                     30,
                                     6])                                    # Value used if joining together skeletons for inner and outer jets

###########################################################################################################
# Parameter used for finding section edges.
###########################################################################################################
# nSig_armss =               np.array([np.array([1.5,4]),
#                                      np.array([1.5,4]),
#                                      np.array([1.5,4])])                    # The multiple of the RMS flux threshold value for each jet arm
nSig_armss =               np.array([np.array([1.0,1.0]),
                                     np.array([1.0,1.0]),
                                     np.array([1.0,1.0])])                    # The multiple of the RMS flux threshold value for each jet arm

###########################################################################################################
# Parameters used for map plotting.
###########################################################################################################
ImFractions  = np.array([0.9,
                         1.0,
                         1.0])                                              # The fraction of the source the final image is cut down to for plotting
vmins        = np.array([0,
                         0,
                         0])                                                # Linear mapping colour range minimum
vmaxs        = np.array([0.010,
                         0.010,
                         0.010])                                            # Linear mapping colour range maximum

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
    global rdel; global ddel
    global beamsize; global beamarea
    global freq
    global sCentre
    global sRA; global sDec
    global sSize
    global bgRMS; global bgMean
    global CutdownSize0; global CutDownSize1
    global GaussSigmaFilter 
    global ContoursLevelPerc
    global MaxRemoveSmallHolesArea
    global MaximumLoopJumpPixels
    global nSig_s
    global SplitInnerOuterSkeleton; global nSig_s_outer; global JoinInterpolatePoints
    global nSig_arms
    global ImFraction; global vmin; global vmax

    # Set the map number
    map_number = current_map_number

    # Initialise all map-specific parameter values for this map number
    map_file = map_files[map_number]
    rdel = rdels[map_number]; ddel = ddels[map_number]
    beamsize = beamsizes[map_number]; beamarea = beamareas[map_number]
    freq = freqs[map_number]
    sCentre = sCentres[map_number]
    sRA = sRAs[map_number]; sDec = sDecs[map_number]
    sSize = sSizes[map_number]
    bgRMS = bgRMSs[map_number]; bgMean = bgMeans[map_number]

    CutdownSize0 = CutdownSize0s[map_number]; CutDownSize1 = CutDownSize1s[map_number]
    GaussSigmaFilter = GaussSigmaFilters[map_number]
    ContoursLevelPerc = ContoursLevelPercs[map_number]
    MaxRemoveSmallHolesArea = MaxRemoveSmallHolesAreas[map_number]
    MaximumLoopJumpPixels = MaximumLoopJumpPixelss[map_number]
    nSig_s = nSig_ss[map_number]
    SplitInnerOuterSkeleton = SplitInnerOuterSkeletons[map_number]; nSig_s_outer = nSig_s_outers[map_number]; JoinInterpolatePoints = JoinInterpolatePointss[map_number]

    nSig_arms = nSig_armss[map_number]

    ImFraction = ImFractions[map_number]; vmin = vmins[map_number]; vmax = vmaxs[map_number]

    print(' ')
    print('--------------------------------------------------------------------------------------')
    print('Starting modelling for ' + sName + ' - map ' + str(map_number+1) + ' (' + map_file + ')')
    print('--------------------------------------------------------------------------------------')
    print(' ')