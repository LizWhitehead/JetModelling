#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetModelling_MapSetup.py
Setup globals for map-specific parameters.
Created by LizWhitehead - 18/04/2025
"""

from math import nan

def setup_map_specific_parameters():
    # Initialise constants specific to the map

    global map_file     ## FITS map file
    global rdel         ## The equivalent pixel value for RA in FITS file in degrees
    global ddel         ## The equivalent pixel value for DEC in FITS file in degrees
    global nSig         ## The multiple of sigma for the RMS reduction
    global beamsize     ## Beam size in arcsecs
    global beamarea     ## Beam area in pixels
    global freq         ## Observation frequency
    global sName        ## source name
    global sRA          ## map centre RA in degrees
    global sDec         ## map centre Dec in degrees
    global sSize        ## source size in pixels (one side of containing square)
    global bgRMS        ## background flux RMS value in Jy/beam
    global rShift       ## red shift
    global sourceR      ## distance to source in Mpc
    global angScale     ## angular scale at source in kpc/arcsec
    global sRadioRA     ## source centre RA in degrees
    global sRadioDec    ## source centre Dec in degrees

    map_file = 'C:/Maps/NGC1044-cutout.fits'
    rdel = -0.000486111111; ddel = 0.000486111111
    nSig = 4
    beamsize = 6.99
    beamarea = 18.12937258110873
    freq = 2.624893188e9
    sName = 'NGC1044'
    sRA = 40.278375
    sDec = 8.733833333
    sSize = 667
    bgRMS = 0.000012
    rShift = 0.021208
    sourceR = 83.40
    angScale = 1.75
    sRadioRA = nan
    sRadioDec = nan
    # sRadioRA = 16.8539791667      # Not used to locate the source centre as not exact enough
    # sRadioDec = 32.4125416667     # Not used to locate the source centre as not exact enough