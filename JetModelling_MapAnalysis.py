#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetModelling_MapAnalysis.py
Map analysis functions.
Created by LizWhitehead - 06/05/2025
"""

import JetModelling_Constants as JMC
import JetModelling_MapSetup as JMS
from astropy.io import fits
from astropy import units as u
from radio_beam import Beam
import numpy as np
from warnings import simplefilter, resetwarnings
from math import floor

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

def GetMaximumFluxAlongLine(flux_array, start_point, end_point):

    """
    Get the position of the maximum flux value along a line.

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
    xmax_coord - x co-ordinate of maximum flux value

    ymax_coord - y co-ordinate of maximum flux value
    """

    # Get the flux distribution along the line
    x_axis_values, flux_values = GetFluxDistributionAlongLine(flux_array, start_point, end_point)

    dist_tot = np.sqrt( (end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2 )     # Total length of the line
    sin_theta = (end_point[1] - start_point[1]) / dist_tot                     # sin of the line incline angle
    cos_theta = (end_point[0] - start_point[0]) / dist_tot                     # cos of the line incline angle

    # Find the position of the maximum flux along the line
    arg_fluxmax = np.nanargmax(flux_values)
    xmax_coord = (x_axis_values[arg_fluxmax] * cos_theta) + start_point[0]
    ymax_coord = (x_axis_values[arg_fluxmax] * sin_theta) + start_point[1]

    return xmax_coord, ymax_coord

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

    # Get limiting percentile flux value
    flux_value_limit = np.max(flux_values) * (100 - JMC.flux_percentage) / 100.0
    if flux_value_limit < 0:
        # Add enough to make flux values positive
        flux_values += abs(np.min(flux_values))
        flux_value_limit = np.max(flux_values) * (100 - JMC.flux_percentage) / 100.0

    # Get min and max x values for defined percentile width
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

def RefineEdgesAlongJetSide(flux_array, edge_points):

    """
    For each edge line along a jet side, use a defined percentile width 
    of the flux distribution to refine the edges.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    edge_points - 2D array, shape(n,7)
                  Points on one side of the jet, corresponding to the 
                  closest distance to that edge from the corresponding ridge point.

    Returns
    -----------
    updated_edge_points - 2D array, shape(n,5)
                          Edge points, updated to be the limits of the flux percentile width

    Notes
    -----------
    """

    # Initialise updated edge points array
    updated_edge_points = np.empty((0,7))

    if JMC.flux_percentage == 100:
        # Edges are returned unchanged
        updated_edge_points = edge_points
    else:
        # Update all edgepoints to be the limits of the flux percentile width
        for x1,y1, x2,y2, R, ridge_x,ridge_y in edge_points:

            start_point = np.array([x1,y1]); end_point = np.array([x2,y2])

            if x1 != x2 or y1 != y2:
                start_point, end_point = RefineEdgesFromFluxPercentileWidth(flux_array, start_point, end_point)

            updated_edge_points = np.vstack((updated_edge_points, np.hstack((start_point, end_point, R, ridge_x, ridge_y)) ))

    return updated_edge_points

#############################################

def RemoveBrightSources(flux_array, source_data):

    """
    Remove bright sources by setting the flux to zero in specified pixels

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    source_data - 2D array, shape(n,3)
                  centre x, centre y, +/- pixel range to remove

    Returns
    -----------
    updated_flux_array - 2D array, shape(n,5)
                         raw image image with bright sources removed

    Notes
    -----------
    """

    # Remove flux in +/- specified number of pixels in x/y directions around the sources' centre
    cutout_offset = GetCutoutOffset(JMS.mCentre, np.array([JMS.sSize[1],JMS.sSize[0]]))
    for centre_x, centre_y, num_pix in source_data.astype('int'):
        if centre_x != -1:
            cutout_centre_x = centre_x - cutout_offset[0]; cutout_centre_y = centre_y - cutout_offset[1]
            xmin = cutout_centre_x - num_pix; xmax = cutout_centre_x + num_pix
            ymin = cutout_centre_y - num_pix; ymax = cutout_centre_y + num_pix
            flux_array[ymin:ymax, xmin:xmax] = 0.0

    return flux_array

#############################################

def GetCutoutOffset(centre_coords, cutout_shape):

    """
    Get the x,y offset of the cutout array relative to the original array

    Parameters
    -----------
    centre_coords - original x,y coordinates of the map centre

    cutout_shape - shape of the cutout

    Returns
    -----------
    cutout_offset - x,y offset of the cutout relative to the original array
    """

    cutout_offset_x = floor(centre_coords[0] - cutout_shape[1]/2.0)
    cutout_offset_y = floor(centre_coords[1] - cutout_shape[0]/2.0)
    cutout_offset = np.array([cutout_offset_x, cutout_offset_y])

    return cutout_offset

#############################################

def PopulateMapData():

    """
    Populates map data from the FITS file header
    """
    simplefilter(action="ignore")

    for map_number in range(JMS.map_count):
        with fits.open(JMS.map_files[map_number]) as map_file_hdu: 
            map_hdr = map_file_hdu[0].header

            the_beam = Beam.from_fits_header(map_hdr)
            bmaj = the_beam.major.to(u.arcsec).value; bmin = the_beam.minor.to(u.arcsec).value

            JMS.rdels[map_number] = abs(map_hdr['CDELT1']); JMS.ddels[map_number] = abs(map_hdr['CDELT2'])
            JMS.sRAs[map_number] = map_hdr['CRVAL1']; JMS.sDecs[map_number] = map_hdr['CRVAL2']
            JMS.mCentres[map_number] = np.array([map_hdr['CRPIX1'], map_hdr['CRPIX2']])
            JMS.freqs[map_number] = map_hdr['CRVAL3']
            JMS.beamsizes[map_number] = bmaj / (JMS.ddels[map_number] * 3600)
            pix_area_arcsec = (JMS.rdels[map_number] * 3600) * (JMS.ddels[map_number] * 3600)
            beamarea_arcsec = bmaj * bmin * (np.pi / (4 * np.log(2))) 
            JMS.beamareas[map_number] = beamarea_arcsec / pix_area_arcsec

            if JMC.use_manual_RMS_and_mean:
                JMS.bgRMSs[map_number] = JMS.bgRMSs_manual[map_number]; JMS.bgMeans[map_number] = JMS.bgMeans_manual[map_number]
            else:
                bgRMS , bgMean  = GetRMSAndMean(map_file_hdu, JMS.mCentres[map_number], JMS.sSizes[map_number])
                JMS.bgRMSs[map_number] = bgRMS; JMS.bgMeans[map_number] = bgMean

    resetwarnings()

#############################################

def GetRMSAndMean(map_file_hdu, mCentre, sSize):

    """
    Returns the background flux RMS and mean values by finding the values in small regions
    around the edge of the rectangle containing the source.

    Parameters
    -----------
    map_file_hdu - map hdu

    mCentre - map centre (x,y) in pixels

    sSize - Source size in pixels (x,y sides of containing rectangle)

    Returns
    -----------
    bgRMS - Background flux RMS value in Jy/beam

    bgMean - Background flux mean pixel value in Jy/beam
    """
    CALC_PTS_PER_SIDE = 4       # Number of points of RMS/mean measurements on one side of containing rectangle
    PERC_REG_SIZE = 5           # Region size as a percentage of the map cutout size

    # Initialise arrays of RMS and means
    all_RMS = np.empty(0); all_mean = np.empty(0)

    # Get cut-out area
    cutout_offset = GetCutoutOffset(mCentre, np.array([sSize[1],sSize[0]]))
    xmin = cutout_offset[0]; xmax = xmin + sSize[0]
    ymin = cutout_offset[1]; ymax = ymin + sSize[1]
    x1y1 = np.array([xmin, ymax]); x2y1 = np.array([xmax, ymax]); x1y2 = np.array([xmin, ymin])

    # Get the RMS and mean flux values around the sides of the enclosing rectangle.
    # For each side, remove the maximum RMS value, in case it is picking up part of the jet or other sources.

    reg_halfsize = floor(np.min(sSize) * (PERC_REG_SIZE/100.) / 2)      # square region half-size
    stepsize_x = float((x2y1[0] - x1y1[0]) / CALC_PTS_PER_SIDE)         # step size in x
    stepsize_y = float((x1y1[1] - x1y2[1]) / CALC_PTS_PER_SIDE)         # step size in y
    image_data = map_file_hdu[0].data.squeeze()                         # image data

    # Along the top and bottom
    side_RMS = np.empty(0); side_mean = np.empty(0)
    for icnt in range(CALC_PTS_PER_SIDE):
        regcentre = np.array([floor(x1y1[0] + (icnt*stepsize_x) + stepsize_x/2.0), floor(x1y1[1] - stepsize_y/2.0)])            # Get region centre
        rms, mean = GetRMSAndMeanForRegion(image_data, regcentre, reg_halfsize)                                                 # Get rms and mean for region
        side_RMS = np.hstack((side_RMS,rms)); side_mean = np.hstack((side_mean,mean))
    arg_maxRMS = np.argmax(side_RMS); side_RMS = np.delete(side_RMS, arg_maxRMS); side_mean = np.delete(side_mean, arg_maxRMS)  # Remove the maximum RMS value
    all_RMS = np.hstack((all_RMS,side_RMS)); all_mean = np.hstack((all_mean,side_mean))

    side_RMS = np.empty(0); side_mean = np.empty(0)
    for icnt in range(CALC_PTS_PER_SIDE):
        regcentre = np.array([floor(x1y2[0] + (icnt*stepsize_x) + stepsize_x/2.0), floor(x1y2[1] + stepsize_y/2.0)])            # Get region centre
        rms, mean = GetRMSAndMeanForRegion(image_data, regcentre, reg_halfsize)                                                 # Get rms and mean for region
        side_RMS = np.hstack((side_RMS,rms)); side_mean = np.hstack((side_mean,mean))
    arg_maxRMS = np.argmax(side_RMS); side_RMS = np.delete(side_RMS, arg_maxRMS); side_mean = np.delete(side_mean, arg_maxRMS)  # Remove the maximum RMS value
    all_RMS = np.hstack((all_RMS,side_RMS)); all_mean = np.hstack((all_mean,side_mean))

    # Down the sides
    side_RMS = np.empty(0); side_mean = np.empty(0)
    for icnt in range(CALC_PTS_PER_SIDE):
        regcentre = np.array([floor(x1y1[0] + stepsize_x/2.0), floor(x1y1[1] - (icnt*stepsize_y) - stepsize_y/2.0)])            # Get region centre
        rms, mean = GetRMSAndMeanForRegion(image_data, regcentre, reg_halfsize)                                                 # Get rms and mean for region
        side_RMS = np.hstack((side_RMS,rms)); side_mean = np.hstack((side_mean,mean))
    arg_maxRMS = np.argmax(side_RMS); side_RMS = np.delete(side_RMS, arg_maxRMS); side_mean = np.delete(side_mean, arg_maxRMS)  # Remove the maximum RMS value
    all_RMS = np.hstack((all_RMS,side_RMS)); all_mean = np.hstack((all_mean,side_mean))

    side_RMS = np.empty(0); side_mean = np.empty(0)
    for icnt in range(CALC_PTS_PER_SIDE):
        regcentre = np.array([floor(x2y1[0] - stepsize_x/2.0), floor(x2y1[1] - (icnt*stepsize_y) - stepsize_y/2.0)])            # Get region centre
        rms, mean = GetRMSAndMeanForRegion(image_data, regcentre, reg_halfsize)                                                 # Get rms and mean for region
        side_RMS = np.hstack((side_RMS,rms)); side_mean = np.hstack((side_mean,mean))
    arg_maxRMS = np.argmax(side_RMS); side_RMS = np.delete(side_RMS, arg_maxRMS); side_mean = np.delete(side_mean, arg_maxRMS)  # Remove the maximum RMS value
    all_RMS = np.hstack((all_RMS,side_RMS)); all_mean = np.hstack((all_mean,side_mean))

    return np.mean(all_RMS), np.mean(all_mean)

#############################################

def GetRMSAndMeanForRegion(image_data, regcentre, reg_halfsize):

    """
    Get the Flux RMS and mean for this region
    """

    xmin = regcentre[0] - reg_halfsize; xmax = regcentre[0] + reg_halfsize
    ymin = regcentre[1] - reg_halfsize; ymax = regcentre[1] + reg_halfsize
    region_data = np.ma.masked_invalid(image_data[ymin:ymax, xmin:xmax])

    rms = np.std(region_data)
    mean = np.mean(region_data)

    return rms, mean
