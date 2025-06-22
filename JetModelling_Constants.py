#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetModelling_Constants.py
Constants for JetModelling project
Created by LizWhitehead - 08/05/2025
"""

debug = True                    # Turns on debug print statements if True
ridgeline_only = False          # Run ridgeline code only

# Ridgelines
ridge_centre_search_points = 6  # Number of ridgepoints on either side of first point to search for source position
ridgelines_from_data = True     # Create ridgelines from existing ridgeline data
ridgelines_from_data_arm1 = 'C:/Maps/NGC1044_ridge1.txt'    # Input data file for arm1
ridgelines_from_data_arm2 = 'C:/Maps/NGC1044_ridge2.txt'    # Input data file for arm2

# Edgepoints
search_angle = 30               # Edgepoint search angle in degrees
MaxRFactor = 100                # Maximum factor for increase of step size before re-initialising edgepoint algorithm 
R = 5                           # Step size along the jet
MinIntpolFactor = 1.5           # Minimum length factor for an edgeline for an edgepoint to be added
MaxIntpolSections = 6           # Maximum number of interpolated points along an edgeline

# Jet Sections
MinSectionsPerArm = 25          # Minimum number of merged sections per arm of the jet
MaxSectionsPerArm = 60          # Maximum number of merged sections per arm of the jet
MaxMergeIterations = 20         # Maximum number of iterations to merge sections to within required number
PercChangeInMaxFlux = 10        # Percentage change in max flux for each merge iteration

# Plotting
ImFraction = 0.5                # The fraction of the source the final image is cut down to for plotting