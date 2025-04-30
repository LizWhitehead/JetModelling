#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetModelling_MapSetup.py
Setup globals fro map-specific parameters.
Created by LizWhitehead - 18/04/2025
"""

def setup_map_specific_parameters():
    # Initialise constants specific to the map

    global map_file ## FITS map file
    global rdel     ## The equivalent pixel value for RA in FITS file in degrees
    global ddel     ## The equivalent pixel value for DEC in FITS file in degrees
    global nSig     ## The multiple of sigma for the RMS reduction
    global beamsize ## Beam size in arcsecs
    global beamarea ## Beam area in pixels
    global freq     ## Observation frequency
    global sName    ## source name
    global sRA      ## source RA
    global sDec     ## source Dec
    global sSize    ## source size in pixels (one side of containing square)
    global bgRMS    ## background flux RMS value in Jy/beam
    global rShift   ## red shift

    map_file = 'C:/Maps/3C31.HGEOM2Copy.FITS'
    rdel = -0.0003611111024; ddel = 0.0003611111024
    nSig = 12.0
    beamsize = 5
    beamarea = 16.76195591180468
    freq = 1.636e9
    sName = '3C31'
    sRA = 16.8217500000
    sDec = 32.4255277778
    sSize = 456
    bgRMS = 0.0002
    rShift = 0.0169