#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetModelling_MapAnalysis.py
Map analysis functions.
Created by LizWhitehead - 06/05/2025
"""

import JetModelling_Constants as JMC
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm, kstest
import matplotlib.pyplot as plt

def GetFluxDistributionAlongLine(flux_array, start_point, end_point):

    """
    Get the flux distribution along a line, at defined pixel intervals.
    Assumes x co-ordinate of end_point > x co-ordinate of start_point

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    start_point - 1D array
                  Co-ordinates of line start point (x,y)

    end_point - 1D array
                Co-ordinates of line end point (x,y)
    
    Constants
    ---------

    Returns
    -----------
    x_axis_values - 1D array
                    Array of pixel distances along the line

    y_axis_values - 1D array
                    Array of flux values at points along the line

    Notes
    -----------
    """
    pixel_interval = 0.1                                # pixel interval for flux values
    x_axis_values = []; y_axis_values = []              # Initialise x and y axis values

    # Fill invalid (nan) values with zeroes in flux_array
    flux_array = np.ma.filled(np.ma.masked_invalid(flux_array), 0)

    # Convert end points to offsets from first end point
    offset = np.array([start_point[0], start_point[1]])
    startpt = np.array([0.0,0.0])
    endpt = end_point - offset 

    dist_tot = np.sqrt( endpt[0]**2 + endpt[1]**2 )     # Total length of the line
    sin_theta = endpt[1] / dist_tot                     # sin of the line incline angle
    cos_theta = endpt[0] / dist_tot                     # cos of the line incline angle

    # Loop along the line until the distance exceeds that of end_point
    dist = 0.0; x_coord = startpt[0]; y_coord = startpt[1]
    try:
        while dist < (dist_tot + pixel_interval):

            if dist > dist_tot: 
                dist = dist_tot; x_coord = endpt[0]; y_coord = endpt[1]      # Reached the line end point

            # Find map pixel co-ordinate containing this point on the line
            map_x_coord = np.floor(x_coord + offset[0]).astype('int')
            map_y_coord = np.floor(y_coord + offset[1]).astype('int')
        
            # Add the x and y values to the arrays
            flux = flux_array[map_y_coord, map_x_coord]
            if not np.isnan(flux):
                x_axis_values.append(dist)
                y_axis_values.append(flux)

            dist += pixel_interval
            x_coord = dist * cos_theta
            y_coord = dist * sin_theta
    except Exception as e:
        print (e)

    return np.array(x_axis_values), np.array(y_axis_values)

#############################################

def GetMinimumFluxAlongLine(flux_array, start_point, end_point):

    """
    Get the position of the minimum flux value along a line.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    start_point - 1D array
                  Co-ordinates of line start point (x,y)

    end_point - 1D array
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
    x_axis_values, flux_values = GetFluxDistributionAlongLine(flux_array, start_point, end_point)

    dist_tot = np.sqrt( (end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2 )     # Total length of the line
    sin_theta = (end_point[1] - start_point[1]) / dist_tot                     # sin of the line incline angle
    cos_theta = (end_point[0] - start_point[0]) / dist_tot                     # cos of the line incline angle

    # Find the position of the minimum flux along the line
    arg_fluxmin = np.nanargmin(flux_values)
    xmin_coord = (x_axis_values[arg_fluxmin] * cos_theta) + start_point[0]
    ymin_coord = (x_axis_values[arg_fluxmin] * sin_theta) + start_point[1]

    return xmin_coord, ymin_coord

#############################################

def RefineEdgesFromFluxPercentileWidth(flux_array, start_point, end_point):

    """
    Find new edge points from the width of a defined percentile of the 
    flux distribution along the line.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    start_point - 1D array
                  Co-ordinates of line start point (x,y)

    end_point - 1D array
                Co-ordinates of line end point (x,y)

    Returns
    -----------
    updated_start_point - 1D array
                          Start point, updated to be the lower limit of the flux percentile width

    updated_end_point - 1D array
                        End point, updated to be the higher limit of the flux percedntile width

    Notes
    -----------
    """

    # Get the flux distribution along the line
    x_axis_values, flux_values = GetFluxDistributionAlongLine(flux_array, start_point, end_point)

    # Get min and max x values for defined percentile width
    flux_value_limit = np.max(flux_values) * (100 - JMC.flux_percentile) / 100.0
    icnt = -1
    for flux in flux_values:                        # Values with increasing x
        icnt += 1
        if flux >= flux_value_limit:
            x_axis_values_min = x_axis_values[icnt]
            break

    icnt = len(flux_values)
    for flux in flux_values[::-1]:                  # Values with decreasing x
        icnt -= 1
        if flux >= flux_value_limit:
            x_axis_values_max = x_axis_values[icnt]
            break

    # Reset back to map co-ordinates
    dist_tot = np.sqrt( (end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2 )     # Total length of the line
    sin_theta = (end_point[1] - start_point[1]) / dist_tot                     # sin of the line incline angle
    cos_theta = (end_point[0] - start_point[0]) / dist_tot                     # cos of the line incline angle
    x_start_point = (x_axis_values_min * cos_theta) + start_point[0]
    y_start_point = (x_axis_values_min * sin_theta) + start_point[1]
    x_end_point = (x_axis_values_max * cos_theta) + start_point[0]
    y_end_point = (x_axis_values_max * sin_theta) + start_point[1]

    updated_start_point = np.array([x_start_point, y_start_point])
    updated_end_point = np.array([x_end_point, y_end_point])

    return updated_start_point, updated_end_point

#############################################

def RefineEdgesAlongJetArm(flux_array, edge_points):

    """
    For each edge line along a jet arm, use a defined percentile width 
    of the flux distribution to refine the edges.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    edge_points - 2D array, shape(n,5)
                  Points on one arm of the jet, corresponding to the 
                  closest distance to that edge from the corresponding ridge point.

    Returns
    -----------
    updated_edge_points - 2D array, shape(n,5)
                          Edge points, updated to be the limits of the flux percentile width

    Notes
    -----------
    """

    # Initialise updated edge points array
    updated_edge_points = np.empty((0,5))

    if JMC.flux_percentile == 100:
        # Edges are returned unchanged
        updated_edge_points = edge_points
    else:
        # Update all edgepoints to be the limits of the flux percentile width
        for x1,y1, x2,y2, R in edge_points:

            start_point = np.array([x1,y1]); end_point = np.array([x2,y2])
            start_point, end_point = RefineEdgesFromFluxPercentileWidth(flux_array, start_point, end_point)

            updated_edge_points = np.vstack((updated_edge_points, np.hstack((start_point, end_point, R)) ))

    return updated_edge_points
