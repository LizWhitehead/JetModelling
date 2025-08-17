#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetModelling_MapSetup.py
Map-specific parameters
Created by LizWhitehead - 18/04/2025
"""

from math import nan

map_file = 'C:/Maps/NGC1044_l_band-cutout.fits'     # FITS map file
rdel = -0.0003055555556; ddel = 0.0003055555556     # Equivalent pixel value for RA/Dec in FITS file in degrees
nSig = 2.0                                          # The multiple of sigma for the RMS reduction
beamsize = 6.99                                     # Beam size in arcsecs
beamarea = 18.12937258110873                        # Beam area in pixels
freq = 1283895507.8125                              # Observation frequency
sName = 'NGC1044'                                   # Source name
sRA = 40.278375                                     # Map centre RA in degrees
sDec = 8.733833333                                  # Map centre Dec in degrees
sSize = 1500                                        # Source size in pixels (one side of containing square)
bgRMS = 0.000006                                    # Background flux RMS value in Jy/beam
sRadioRA = nan                                      # Source (radio) centre RA in degrees
sRadioDec = nan                                     # Source (radio) centre Dec in degrees
# sRadioRA = 16.8539791667      # Not used to locate the source centre as not exact enough
# sRadioDec = 32.4125416667     # Not used to locate the source centre as not exact enough