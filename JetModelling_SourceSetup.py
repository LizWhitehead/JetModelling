#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetModelling_SourceSetup.py
Source-specific parameters
Created by LizWhitehead - 25/07/2025
"""

import numpy as np

rShift = 0.0169         # Red shift of source
sourceR = 73.3          # Distance to source in Mpc
angScale = 0.3438       # Angular scale at source in kpc/arcsec
spectral_index = 0.55   # Spectral index
emin = 8e-13            # electron energy lower cutoff
emax = 8e-8             # electron energy higher cutoff
logkmax = 4.0           # maximum allowed kappa

#########################################################################
# Environmental parameters.
# 2D arrays: [dist along jet (kpc), parameter value]
#########################################################################

# External pressure (Pa)
ext_pressure_arm1 = np.array([[1.812,1.069e-11], [2.977,5.715e-12], [4.207,3.974e-12], [5.502,3.06e-12], [6.472,2.368e-12], \
                              [8.091,1.907e-12], [9.061,1.594e-12], [10.356,1.381e-12], [11.650,1.228e-12], [12.945,1.117e-12], \
                              [18.447,8.671e-13], [21.036,8.161e-13], [23.948,7.827e-13], [26.861,7.516e-13], [29.450,7.206e-13], \
                              [32.039,6.942e-13], [42.071,6.09e-13], [55.016,5.886e-13], [67.961,4.765e-13], [84.142,3.976e-13], \
                              [100.324,3.319e-13], [136.4,2.341e-13]])

ext_pressure_arm2 = np.array([[3.204,5.283e-12], [5.502,3.131e-12], [7.767,2.008e-12], [10.032,1.461e-12], [12.298,1.175e-12], \
                              [14.563,1.012e-12], [16.828,9.08e-13], [19.094,8.329e-13], [21.036,7.826e-13], [23.625,7.433e-13], \
                              [25.890,7.09e-13], [28.479,6.802e-13], [31.068,6.569e-13], [32.362,6.34e-13], [35.599,6.119e-13], \
                              [45.307,5.56e-13], [51.780,5.341e-13], [55.016,5.144e-13], [67.961,4.595e-13], [71.197,4.396e-13], \
                              [74.434,4.215e-13], [116.9,2.46e-13]])
