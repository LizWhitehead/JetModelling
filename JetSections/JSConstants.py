#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSConstants.py
Constants for jet sections processing.
Created by LizWhitehead - 01/05/2025
"""

##  Debugging on or off turns on print statements if True
debug = True

## FindEdgePoints
MaxRFactor = 3  ## Maximum factor for increase of step size before re-initialising edgepoint algorithm 
MinIntpolFactor = 1.5    ## Minimum length factor for an edgeline for an edgepoint to be added
MaxIntpolSections = 6    ## Maximum number of interpolated points along an edgeline

## Jet Sections
MinSectionsPerArm = 18   ## Minimum number of merged sections per arm of the jet
MaxSectionsPerArm = 25   ## Maximum number of merged sections per arm of the jet
MaxMergeIterations = 10  ## Maximum number of iterations to merge sections to within required number
PercChangeInMaxFlux = 10 ## Percentage change in max flux for each merge iteration
