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
geo_lentol = 0.2                # Tolerance for calculation of geometric lengths (pixels)
geo_angtol = 0.01               # Tolerance for calculation of geometric angles (radians)
nSig = np.array([4,4])                   # The sigma of the RMS flux threshold value

# Ridgelines
ridgeline_method = RidgelineMethod.FROMDATA  # Method for creating ridgelines
ridge_centre_search_points = 7  # Number of ridgepoints on either side of first point to search for source centre
R = 10                         # Maximum step size of ridgeline in pixels

# Ridgelines (RL-Xid)
R_rl = 4                       # Step size of ridgeline in pixels
dphi = 60                       # Half angle of search cone

# Edgepoints
search_angle = 5                # Edgepoint search angle in degrees
MaxRFactor = 100                # Maximum factor for increase of step size before re-initialising edgepoint algorithm 
R_es = 5                      # Step size along the jet in pixels
SearchRadiusIncFactor = 1.25  # Factor to increase the edgepoint search radius for each ridgepoint
NumRetryAttempts=3 #Retries if edge on search radius
MinIntpolFactor = 4             # Minimum length factor for an edgeline for an edgepoint to be added
MaxIntpolSections = 6           # Maximum number of interpolated points along an edgeline
flux_percentage = 95            # Flux percentile limit for refining jet edges (100 => no change)

# Jet Sections
MergeMinDeltaRFactor = 1.5      # Multiplication factor of minimum delta of merge R (>= 1)
MergeRIncreaseStep = 1.15       # Step increase in delta of merge R, in pixels (>=1)

# Regions
x_offset = 0.0                  # x offset of the region co-ordinates to full image co-ordinates in pixels
y_offset = 0.0                  # y offset of the region co-ordinates to full image co-ordinates in pixels
max_vertices = 2000             # maximum number of vertices in a region polygon

# Plotting
flux_factor = 1               # Factor to multiply flux by for plotting
