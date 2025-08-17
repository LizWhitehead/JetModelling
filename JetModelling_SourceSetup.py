#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetModelling_SourceSetup.py
Source-specific parameters
Created by LizWhitehead - 25/07/2025
"""

import numpy as np

rShift = 0.021208       # Red shift of source
sourceR = 83.40         # Distance to source in Mpc
angScale = 1.75         # Angular scale at source in kpc/arcsec
spectral_index = 0.55   # Spectral index
emin = 8e-13            # electron energy lower cutoff
emax = 8e-8             # electron energy higher cutoff
logkmax = 4.0           # maximum allowed kappa

#########################################################################
# Environmental parameters.
# 2D arrays: [dist along jet (kpc), parameter value]
#########################################################################

# External pressure (Pa)
ext_pressure_arm1 = np.nan
ext_pressure_arm2 = np.nan
