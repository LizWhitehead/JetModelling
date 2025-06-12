#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetModelling_MapAnalysis.py
Map analysis functions.
Created by LizWhitehead - 06/05/2025
"""

import numpy as np

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

    # Fill invalid (nan) values with zeroes in flux_array
    flux_array = np.ma.filled(np.ma.masked_invalid(flux_array), 0)

    # Convert end points to offsets from first end point
    offset = np.array([start_point[0], start_point[1]])
    startpt = np.array([0.0,0.0])
    endpt = end_point - offset 

    # Gradient of line
    line_grad = (endpt[1] - startpt[1]) / (endpt[0] - startpt[0])

    # Determine x intervals for 0.5 pixel intervals along the line
    x_interval = np.cos(np.arctan2((endpt[1] - startpt[1]), (endpt[0] - startpt[0]))) * 0.5

    # Loop along the line until x exceeds that of end_point
    x_coord = startpt[0]; y_coord = startpt[1]
    try:
        while x_coord < (endpt[0] + x_interval):

            if x_coord > endpt[0]: 
                x_coord = endpt[0]; y_coord = endpt[1]      # Reached the line end point

            # Find map pixel co-ordinate containing this point on the line
            map_x_coord = np.floor(x_coord + offset[0]).astype('int')
            map_y_coord = np.floor(y_coord + offset[1]).astype('int')
        
            # Add the x and y values to the arrays
            flux = flux_array[map_y_coord, map_x_coord]
            if not np.isnan(flux):
                x_values.append(x_coord)
                y_values.append(flux)

            x_coord += x_interval
            y_coord = line_grad * x_coord
    except Exception as e:
        print (e)

    return np.array(x_values), np.array(y_values)

#############################################

def GetMinimumFluxAlongLine(flux_array, start_point, end_point):

    """
    Get the position of the minimum flux value along a line.
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
    xmin_coord - x co-ordinate of minimum flux value

    ymin_coord - y co-ordinate of minimum flux value

    Notes
    -----------
    """
    # Get the flux distribution along the line
    x_values, flux_values = GetFluxDistributionAlongLine(flux_array, start_point, end_point)

    # Gradient of line
    line_grad = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])

    # Find the position of the minimum flux along the line
    arg_fluxmin = np.nanargmin(flux_values)
    xmin_coord = x_values[arg_fluxmin] + start_point[0]
    ymin_coord = (line_grad * x_values[arg_fluxmin]) + start_point[1]

    return xmin_coord, ymin_coord

#############################################

