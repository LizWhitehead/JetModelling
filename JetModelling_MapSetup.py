#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetModelling_MapSetup.py
Map-specific parameters
Created by LizWhitehead - 18/04/2025
"""

from math import nan

map_file = 'C:/Maps/3C31.HGEOM2Copy.FITS'           # FITS map file
rdel = -0.0003611111024; ddel = 0.0003611111024     # Equivalent pixel value for RA/Dec in FITS file in degrees
beamsize = 5                                        # Beam size in arcsecs
beamarea = 16.76195591180468                        # Beam area in pixels
freq = 1.636e9                                      # Observation frequency
sName = '3C31'                                      # Source name
sRA = 16.8217500000                                 # Map centre RA in degrees
sDec = 32.4255277778                                # Map centre Dec in degrees
sSize = 456                                         # Source size in pixels (one side of containing square)
bgRMS = 0.0002                                      # Background flux RMS value in Jy/beam
sRadioRA = nan                                      # Source (radio) centre RA in degrees
sRadioDec = nan                                     # Source (radio) centre Dec in degrees
# sRadioRA = 16.8539791667      # Not used to locate the source centre as not exact enough
# sRadioDec = 32.4125416667     # Not used to locate the source centre as not exact enough