#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetModelling_MapSetup.py
Map-specific parameters
Created by LizWhitehead - 18/04/2025
"""

from math import nan

map_file = 'C:/Maps/NGC1044-cutout.fits'            # FITS map file
rdel = -0.000486111111; ddel = 0.000486111111       # Equivalent pixel value for RA/Dec in FITS file in degrees
nSig = 4                                            # The multiple of sigma for the RMS reduction
beamsize = 6.99                                     # Beam size in arcsecs
beamarea = 18.12937258110873                        # Beam area in pixels
freq = 2.624893188e9                                # Observation frequency
sName = 'NGC1044'                                   # Source name
sRA = 40.278375                                     # Map centre RA in degrees
sDec = 8.733833333                                  # Map centre Dec in degrees
sSize = 667                                         # Source size in pixels (one side of containing square)
bgRMS = 0.000012                                    # Background flux RMS value in Jy/beam
rShift = 0.021208                                   # Red shift of source
sourceR = 83.40                                     # Distance to source in Mpc
angScale = 1.75                                     # Angular scale at source in kpc/arcsec
sRadioRA = nan                                      # Source (radio) centre RA in degrees
sRadioDec = nan                                     # Source (radio) centre Dec in degrees
# sRadioRA = 16.8539791667      # Not used to locate the source centre as not exact enough
# sRadioDec = 32.4125416667     # Not used to locate the source centre as not exact enough