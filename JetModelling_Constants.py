#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetModelling_Constants.py
Constants for JetModelling project
Created by LizWhitehead - 08/05/2025
"""

debug = True                    # Turns on debug print statements if True
ridgeline_only = False          # Run ridgeline code only

# Ridgelines (general)
ridgelines_from_data = True     # Create ridgelines from existing ridgeline data
ridge_centre_search_points = 6  # Number of ridgepoints on either side of first point to search for source position

# Ridgelines (RL-Xid)
R_rl = 5                        # Step size of ridgeline in pixels
dphi = 60                       # Half angle of search cone

# Ridgelines (from data)
ridgelines_from_data_arm1 = 'C:/Maps/NGC1044_ridge1.txt'    # Input data file for arm1
ridgelines_from_data_arm2 = 'C:/Maps/NGC1044_ridge2.txt'    # Input data file for arm2
R_fd = 6                                                    # Maximum step size of ridgeline in pixels

# Edgepoints
search_angle = 30               # Edgepoint search angle in degrees
MaxRFactor = 100                # Maximum factor for increase of step size before re-initialising edgepoint algorithm 
R_es = 5                        # Step size along the jet in pixels
MinIntpolFactor = 1.5           # Minimum length factor for an edgeline for an edgepoint to be added
MaxIntpolSections = 6           # Maximum number of interpolated points along an edgeline
flux_percentile = 100           # Flux percentile limit for refining jet edges (100 => no change)

# Jet Sections
MinSectionsPerArm = 25          # Minimum number of merged sections per arm of the jet
MaxSectionsPerArm = 60          # Maximum number of merged sections per arm of the jet
MaxMergeIterations = 20         # Maximum number of iterations to merge sections to within required number
PercChangeInMaxFlux = 10        # Percentage change in max flux for each merge iteration

# Regions
x_offset = 0                    # x offset of the region co-ordinates to full image co-ordinates in pixels
y_offset = 0                    # y offset of the region co-ordinates to full image co-ordinates in pixels

# Plotting
ImFraction = 0.5                # The fraction of the source the final image is cut down to for plotting
flux_factor = 1                 # Factor to multiply flux by for plotting
vmin = 0                        # Linear mapping colour range minimum
vmax = 0.002                    # Linear mapping colour range maximum