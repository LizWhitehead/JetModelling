#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetModelling_Constants.py
Constants for JetModelling project
Created by LizWhitehead - 08/05/2025

Note: The first jet arm (North) is considered to be that with the highest Y values.
"""

import numpy as np
from enum import Enum

class RidgelineMethod(Enum):
    RLXID = 1
    FROMDATA = 2
    SKELETONIZE = 3

debug = True                    # Turns on debug print statements if True
ridgeline_only = False          # Run ridgeline code only
eqtol = 0.01                    # Tolerance for testing equality of float values

# Ridgelines (general)
ridgeline_method = RidgelineMethod.SKELETONIZE  # Method for creating ridgelines
ridge_centre_search_points = 7  # Number of ridgepoints on either side of first point to search for source position

# Ridgelines (RL-Xid)
R_rl = 5                        # Step size of ridgeline in pixels
dphi = 60                       # Half angle of search cone
nSig = 13.5                     # The multiple of RMS flux threshold value

# Ridgelines (from data)
ridgelines_from_data_arm1 = 'C:/JetModelling_FromData/3C31_ridge_coords1.txt'    # Input data file for arm1
ridgelines_from_data_arm2 = 'C:/JetModelling_FromData/3C31_ridge_coords2.txt'    # Input data file for arm2
R_fd = 6                        # Maximum step size of ridgeline in pixels

# Ridgelines (skeletonize)
R_s = 6                         # Maximum step size of ridgeline in pixels

# Edgepoints
search_angle = 30               # Edgepoint search angle in degrees
MaxRFactor = 100                # Maximum factor for increase of step size before re-initialising edgepoint algorithm 
R_es = 5                        # Step size along the jet in pixels
MaskPreviousEdgepoint = np.array([False,True])  # Flag for each jet arm to mask the previous edgepoint when finding the next edgepoint
MinIntpolFactor = 2.0           # Minimum length factor for an edgeline for an edgepoint to be added
MaxIntpolSections = 4           # Maximum number of interpolated points along an edgeline
flux_percentile = 100           # Flux percentile limit for refining jet edges (100 => no change)

# Jet Sections
MergeMinDeltaRFactor = 1.5      # Multiplication factor of minimum delta of merge R (>= 1)
MergeRIncreaseFactor = 1.15     # Multiplication factor of step increase in delta of merge R (>=1)

# Regions
x_offset = 0.0                  # x offset of the region co-ordinates to full image co-ordinates in pixels
y_offset = 0.0                  # y offset of the region co-ordinates to full image co-ordinates in pixels
max_vertices = 2000             # maximum number of vertices in a region polygon

# Plotting
flux_factor = 1                 # Factor to multiply flux by for plotting
