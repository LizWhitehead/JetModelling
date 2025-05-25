#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSMapAnalysis.py
Map analysis functions.
Created by LizWhitehead - 06/05/2025
"""

from math import isnan
import numpy as np
import time

def GetFluxDistributionAlongLine(flux_array, start_point, end_point):

    """
    Get the flux distribution along a line, at intervals of 0.5 pixels.
    Assumes x co-ordinate of end_point > x co-ordinate of start_point

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    start_point - 1D array, shape (2)
                  Co-ordinates of line start point (x,y)

    end_point - 1D array, shape (2)
                Co-ordinates of line end point (x,y)
    
    Constants
    ---------

    Returns
    -----------
    x_values - 1D array, shape (n)
               Array of pixel distances along the line

    y_values - 1D array, shape (n)
               Array of flux values at x points along the line

    Notes
    -----------
    """
    x_values = []; y_values = []      # Initialise x and y values

    # Convert end points to offsets from first end point
    offset = np.array([start_point[0], start_point[1]])
    start_point = np.array([0.0,0.0])
    end_point -= offset 

    # Gradient of line
    line_grad = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])

    # Determine x intervals for 0.5 pixel intervals along the line
    x_interval = np.cos(np.arctan2((end_point[1] - start_point[1]), (end_point[0] - start_point[0]))) * 0.5

    # Loop along the line until x exceeds that of end_point
    x_coord = start_point[0]; y_coord = start_point[1]
    while x_coord < (end_point[0] + x_interval):

        if x_coord > end_point[0]: 
            x_coord = end_point[0]; y_coord = end_point[1]      # Reached the line end point

        # Find map pixel co-ordinate containing this point on the line
        map_x_coord = np.floor(x_coord + offset).astype('int')
        map_y_coord = np.floor(y_coord + offset).astype('int')
        
        # Add the x and y values to the arrays
        flux = flux_array[map_y_coord, map_x_coord]
        if not np.isnan(flux):
            x_values.append(x_coord)
            y_values.append(flux)

        x_coord += x_interval
        y_coord += line_grad * x_coord

    return np.array(x_values), np.array(x_values)
