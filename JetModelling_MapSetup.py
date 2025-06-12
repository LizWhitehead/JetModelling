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

    map_file = 'C:/Maps/3C31.HGEOM2Copy.FITS'
    rdel = -0.0003611111024; ddel = 0.0003611111024
    nSig = 13.5
    beamsize = 5
    beamarea = 16.76195591180468
    freq = 1.636e9
    sName = '3C31'
    sRA = 16.8217500000
    sDec = 32.4255277778
    sSize = 456
    bgRMS = 0.0002
    rShift = 0.0169
    sourceR = 73.3
    angScale = 0.3438
    sRadioRA = nan
    sRadioDec = nan
    # sRadioRA = 16.8539791667
    # sRadioDec = 32.4125416667