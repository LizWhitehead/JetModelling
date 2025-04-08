#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JPConstants.py
Constants for jet parameters computation.
Created by LizWhitehead - 08/04/2025
"""

##  Debugging on or off turns on print statements if True
debug = True

## Cosmological Constants
H0 = 70                 # Hubble constants in km/s/Mpc
SLight = 299792458      # Speed of light in m/s

def init_maptype_specific_constants(map_type):
    # Initialise constants specific to the map type

    global rdel     ## The equivalent pixel value for RA in FITS file in degrees
    global ddel     ## The equivalent pixel value for DEC in FITS file in degrees
    global beamarea ## Beam area in pixels

    if map_type == 'VLA':
        rdel = -0.0003611111024; ddel = 0.0003611111024
        beamarea = 16.76195591180468
    elif map_type == 'LOFAR-DR1':
        rdel = -0.0004166667; ddel = 0.0004166667
        beamarea = 16.76195591180468
    elif map_type == 'LOFAR-DR2':
        rdel = -0.0004166667; ddel = 0.0004166667
        beamarea = 16.76195591180468
    elif map_type == 'MEERKAT':
        rdel = -0.000305556; ddel = 0.000305556
        beamarea = 16.76195591180468
    else:
        # VLA
        rdel = -0.0003611111024; ddel = 0.0003611111024
        beamarea = 16.76195591180468
