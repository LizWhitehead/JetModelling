#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetModelling_MapAnalysis.py
Map analysis functions.
Created by LizWhitehead - 06/05/2025
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm, kstest
import matplotlib.pyplot as plt

def GetFluxDistributionAlongLine(flux_array, start_point, end_point):

    """
    Get the flux distribution along a line, at intervals of 0.5 pixels.
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
    x_values - 1D array
               Array of pixel distances along the line

    y_values - 1D array
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

    # Ensure x co-ordinate of end_point > x co-ordinate of start_point
    if start_point[0] > end_point[0]:
        temp_end_point = end_point; end_point = start_point; start_point = temp_end_point

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

def FitGaussianToFluxAlongLine(flux_array, start_point, end_point, arm_number, edge_line_number):

    """
    Fit a gaussian to the flux distribution along a line

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

    Notes
    -----------
    """

    # Ensure x co-ordinate of end_point > x co-ordinate of start_point
    if start_point[0] > end_point[0]:
        temp_end_point = end_point; end_point = start_point; start_point = temp_end_point

    # Get the flux distribution along the line
    x_values, flux_values = GetFluxDistributionAlongLine(flux_array, start_point, end_point)

    # Fit a Gaussian to the data
    mean = sum(x_values * flux_values)
    sigma = sum(flux_values * (x_values - mean)**2)
    popt, pcov = curve_fit(Gauss, x_values, flux_values, p0 = [1, mean, sigma])

    x_mean = popt[1]
    x_sigma = abs(popt[2])
    FWHM = 2.3548200 * x_sigma      # 2 * sqrt(2 * ln(2)) * sigma
    x_min = x_mean - (FWHM / 2)
    x_max = x_mean + (FWHM / 2)

    plt.plot(x_values, flux_values, 'o', label='data')
    plt.plot(x_values, Gauss(x_values, *popt), label='fit')
    plt.legend()

    import os
    plt.savefig(os.getcwd() + '/sections/FluxDistribution_' + str(arm_number) + '_' + str(edge_line_number))
    plt.close()

    #kstestResult = kstest(flux_values, lambda x: norm.cdf(x, loc=x_mean, scale=x_sigma))
    kstestResult = kstest(flux_values, 'norm', args=(x_mean, x_sigma))

    return

# Define the Gaussian function
def Gauss(x, a, x0, sigma):
    y = a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    return y

#############################################

def FitGaussianAlongJetArm(flux_array, edge_points, arm_number):

    """
    Fit a gaussian to edge lines along a jet arm

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    edge_points - 2D array, shape(n,4)
                  Points on one arm of the jet, corresponding to the 
                  closest distance to that edge from the corresponding ridge point.

    Returns
    -----------

    Notes
    -----------
    """

    icnt = 0
    for x1,y1, x2,y2, R in edge_points:
        icnt += 1

        if icnt == 1 or icnt % 10 == 0:
            start_point = np.array([x1,y1]); end_point = np.array([x2,y2])
            FitGaussianToFluxAlongLine(flux_array, start_point, end_point, arm_number, icnt)
