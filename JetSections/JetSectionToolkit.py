#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetSectionToolkit.py
Toolkit for sections processing
Created by LizWhitehead - Jan 2025
"""

import JetModelling_MapSetup as JMS
import JetSections.JetSectionFiles as JSF
import JetModelling_Constants as JMC
import JetModelling_MapAnalysis as JMA
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.draw import polygon2mask
import numpy as np
from numpy import pi, sin, cos, dot
from math import tan, atan2, atan, pow, nan, isnan, floor
from shapely.geometry import LineString, Point, Polygon
from regions import PixCoord, PolygonPixelRegion, Regions
import copy

def GetEdgepointsAndSections(flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2):

    """
    Divide both arms of the jet into sections by finding edge points.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    ridge1 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for one arm of the jet
    
    ridge2 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for other arm of the jet

    phi_val1 - 1D array of ridgeline angles for each ridgepoint on
               one arm of the jet
    
    phi_val2 - 1D array of ridgeline angles for each ridgepoint on
               other arm of the jet

    Rlen1 - 1D array of distance from source for each ridgepoint on
            one arm of the jet
    
    Rlen2 - 1D array of distance from source for each ridgepoint on
            other arm of the jet
    
    Constants
    ---------

    Returns
    -----------
    section_parameters1 - 2D array, shape(n,12)
                          Array with section points (x,y * 4), distance from source,
                          flux, volume and area for one arm of the jet

    section_parameters2 - 2D array, shape(n,12)
                          Array with section points (x,y * 4), distance from source
                          flux, volume and area for other arm of the jet

    Notes
    -----------
    """

    # Loop through the ridge points and find the corresponding edgepoints
    if not np.all(np.isnan(ridge1)) and not np.all(np.isnan(ridge2)):

        print('Finding edge points')

        # Find initial edge points
        init_edge_points1 = FindInitEdgePoints(flux_array, np.array(ridge1[0]), phi_val1[0], Rlen1[0])

        # The initial edge points were ordered relative to the ridge phi for the first arm. Re-order for the second arm.
        init_edge_points2 = np.array([init_edge_points1[2], init_edge_points1[3], \
                                      init_edge_points1[0], init_edge_points1[1], init_edge_points1[4], init_edge_points1[5], init_edge_points1[6]])

        # Find all edgepoints for each arm of the jet
        edge_points1 = FindAllEdgePointsForJetArm(flux_array, init_edge_points1, ridge1, phi_val1, Rlen1)
        edge_points2 = FindAllEdgePointsForJetArm(flux_array, init_edge_points2, ridge2, phi_val2, Rlen2)

        # Interpolate extra edge points at points in the jet where significant flux or area (=> calculated volume) is cut off.
        # This will ensure that summed flux and volume are correct when merging segments.
        print('Adding extra edge points')
        edge_points1 = AddEdgePoints(flux_array, edge_points1)
        edge_points2 = AddEdgePoints(flux_array, edge_points2)

        # Refine the edges using a percentile of the flux distribution
        edge_points1 = JMA.RefineEdgesAlongJetArm(flux_array, edge_points1)
        edge_points2 = JMA.RefineEdgesAlongJetArm(flux_array, edge_points2)

        # Get sections and section parameters (distance from source, flux, volume) of the jet
        section_parameters1, section_parameters2, section_perimeters1, section_perimeters2 = \
                             GetJetSections(flux_array, edge_points1, edge_points2, ridge1, ridge2)

        # Plot edgepoint and section data
        PlotEdgePointsAndSections(flux_array, JMS.sName, edge_points1, edge_points2, section_parameters1, section_parameters2)
        
        # Calculate the distance of the section centres along the jet
        section_parameters1 = Calculate_R(section_parameters1)
        section_parameters2 = Calculate_R(section_parameters2)

        # Save files
        SaveEdgepointAndSectionFiles(JMS.sName, edge_points1, edge_points2, section_parameters1, section_parameters2, section_perimeters1, section_perimeters2)
        SaveRegions(JMS.sName, section_perimeters1, section_perimeters2)

    return section_parameters1, section_parameters2

#############################################

def FindAllEdgePointsForJetArm(flux_array, init_edge_points, ridge, phi_val, Rlen):

    """
    Find all edge points for one arm of the jet

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    init_edge_points - 1D array
                       Points on either side of the jet, corresponding to the 
                       closest distance to that edge from the initial ridge point,
                       and their distance from source.
                       (x1,y1,x2,y2,R,ridge_x,ridge_y)

    ridge - 2D array, shape(n,2)
            Array of ridgepoint co-ordinates for one arm of the jet

    phi_val - 1D array of ridgeline angles for each ridgepoint on
              one arm of the jet

    Rlen - 1D array of distance from source for each ridgepoint on
            one arm of the jet
    
    Constants
    ---------

    Returns
    -----------
    edge_points - 2D array, shape(n,7)
                  Points on either side of the jet, corresponding to the 
                  closest distance to that edge from the initial ridge point,
                  and their distance from source.
                  (x1,y1,x2,y2,R,ridge_x,ridge_y)

    Notes
    -----------
    """

    edge_points = np.empty((0,7))                               # Initialise edge point array
    edge_points = np.vstack((edge_points, init_edge_points))    # Add the initial edge points

    # Loop through the ridge points and find the corresponding edgepoints
    if not np.all(np.isnan(ridge)):

        # Find edgepoints for one arm of the jet
        ridge_count = 1; ridge_total = len(ridge)
        while ridge_count < ridge_total:

            if np.isnan(phi_val[ridge_count]):
                break

            if (Rlen[ridge_count] - Rlen[ridge_count-1]) < (JMC.MaxRFactor * JMC.R_es):  # Test whether the last step size has increased by too much
                new_edge_points = FindEdgePoints(flux_array, ridge[ridge_count], phi_val[ridge_count], Rlen[ridge_count], prev_edge_points = edge_points[-1])
            else:
                # Re-initialise edge points algorithm
                new_edge_points = FindInitEdgePoints(flux_array, ridge[ridge_count], phi_val[ridge_count], Rlen[ridge_count])

            if not np.isnan(new_edge_points).any(): 
                edge_points = np.vstack((edge_points, new_edge_points))

            ridge_count += 1

    return edge_points

#############################################

def FindEdgePoints(flux_array, ridge_point, ridge_phi, ridge_R, prev_edge_points):

    """
    Returns edge points, on either side of the jet, corresponding to
    the closest distance to that edge from the supplied ridge point.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    ridge_point - 1D array
                  ridge point co-ordinates along the ridge line

    ridge_phi - float,
                angular direction of the ridge line

    ridge_R - float,
              distance from the source along the jet in pixels

    prev_edge_points - 1D array
                       edge points corresponding to
                       previous ridge point and distance from source
                       (x1,y1,x2,y2,R,ridge_x,ridge_y)
    
    Constants
    ---------

    Returns
    -----------
    edge_points - 1D array
                  Points on either side of the jet, corresponding to the 
                  closest distance to that edge from the corresponding ridge point,
                  and distance from source
                  (x1,y1,x2,y2,R,ridge_x,ridge_y)

    Notes
    -----------
    """
    # Initialise flags
    invalid_edge_point = False; moving_backwards = False

    # Initialise edge_points array
    edge_points = np.full((1,7), np.nan)

    # Initialise the search angle (radians) from the previous edge point
    search_angle = JMC.search_angle * pi/180

    # Fill invalid (nan) values with zeroes in flux_array
    flux_array = np.ma.filled(np.ma.masked_invalid(flux_array), 0)

    # Mask the flux values below the RMS value (JMC.nSig * rms)
    rms_mask = np.ma.masked_where(flux_array <= (JMC.nSig * JMS.bgRMS), flux_array).mask

    # Get polar coordinates for flux_array, around the ridge point
    r, phi = PolarCoordinates(flux_array, ridge_point)

    # Find the r and phi values of the previous edge points
    prev_edge_pix = np.floor(prev_edge_points).astype('int')
    phi_prev1 = phi[prev_edge_pix[1], prev_edge_pix[0]]; phi_prev2 = phi[prev_edge_pix[3], prev_edge_pix[2]]

    # Find the new search angle range on each side of the jet
    phi_min1, phi_max1, phi_min2, phi_max2 = FindSearchAngleRanges(phi_prev1, phi_prev2, ridge_phi)

    # Set the search radius, from half the previous edgepoint to edgepoint distance
    r_prev = np.sqrt( (prev_edge_points[2] - prev_edge_points[0])**2 + (prev_edge_points[3] - prev_edge_points[1])**2 )
    r_radius1 = (r_prev / 2.0) * JMC.SearchRadiusIncFactor; r_radius2 = (r_prev / 2.0) * JMC.SearchRadiusIncFactor 

    # Search for the edge points
    latest_edge_points = GetEdgepointsInRadius(flux_array, r, phi, rms_mask, prev_edge_pix, r_radius1, r_radius2, phi_min1, phi_max1, phi_min2, phi_max2)
    if not np.isnan(latest_edge_points).any(): 

        # Check whether we have moved along too many ridge points since the last valid edge points
        ridge_dist = np.sqrt( (prev_edge_points[5] - ridge_point[0])**2 + (prev_edge_points[6] - ridge_point[1])**2 )
        if (ridge_dist / JMC.R) > 3:

            # If an edge point is exactly at the radius, extend the radius up to 3 times, until the edge point is found within the radius
            search_cnt = 1
            while search_cnt <= 3:
                latest_edge_pix = np.floor(latest_edge_points).astype('int')
                r_edgepoint1 = r[latest_edge_pix[1],latest_edge_pix[0]]; r_edgepoint2 = r[latest_edge_pix[3],latest_edge_pix[2]]
                if (abs(r_edgepoint1 - r_radius1) < JMC.geo_lentol) or (abs(r_edgepoint2 - r_radius2) < JMC.geo_lentol):

                    # One or both of the edge points is exactly on the search radius. Extend the radius of search on that side of the jet and search again.
                    if abs(r_edgepoint1 - r_radius1) < JMC.geo_lentol: r_radius1 = r_radius1 * JMC.SearchRadiusIncFactor
                    if abs(r_edgepoint2 - r_radius2) < JMC.geo_lentol: r_radius2 = r_radius2 * JMC.SearchRadiusIncFactor
                    new_edge_points = GetEdgepointsInRadius(flux_array, r, phi, rms_mask, prev_edge_pix, r_radius1, r_radius2, phi_min1, phi_max1, phi_min2, phi_max2)
                    if not np.isnan(new_edge_points).any():
                        latest_edge_points = new_edge_points
                    else:
                        break
                else:
                    break
                search_cnt += 1  
                
            if not np.isnan(latest_edge_points).any():
                edge_points = np.array([latest_edge_points[0], latest_edge_points[1], latest_edge_points[2], latest_edge_points[3], ridge_R, ridge_point[0], ridge_point[1]])
        else:
            edge_points = np.array([latest_edge_points[0], latest_edge_points[1], latest_edge_points[2], latest_edge_points[3], ridge_R, ridge_point[0], ridge_point[1]])

    return edge_points

#############################################

def GetEdgepointsInRadius(flux_array, r, phi, rms_mask, prev_edge_pix, r_radius1, r_radius2, phi_min1, phi_max1, phi_min2, phi_max2):

    """
    Search for edge points with a defined angle and radius.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    r, phi - polar co-ordinates around the ridge point

    rms_mask - mask where flux is below the rms value

    previous_edge_pix - 1D array
                        previous edge points as integers
                        (x1,y1,x2,y2,R,ridge_x,ridge_y)

    r_radius1, r_radius2 - search radii on each side of the jet

    phi_min1, phi_max1, phi_min2, phi_max2 - min/max search angles on each side of the jet

    Returns
    -----------
    edge_points - 1D array
                  Points on either side of the jet, corresponding to the 
                  closest distance to that edge from the corresponding ridge point
                  (x1,y1,x2,y2)

    Notes
    -----------
    """

    # Initialise flags
    invalid_edge_point = False; moving_backwards = False

    # Initialise edge_points array
    edge_points = np.full((1,4), np.nan)

    # Create a mask to search for nearest edge point on this side of the ridge point
    min_phi = min(phi_min1, phi_max1); max_phi = max(phi_min1, phi_max1)
    quad_min = CheckQuadrant(min_phi); quad_max = CheckQuadrant(max_phi)
    diff = max_phi - min_phi
    if 3 <= quad_min <= 4 and 1 <= quad_max <= 2 and diff > pi:
        phi_mask = np.ma.masked_inside(phi, phi_min1, phi_max1, copy=True).mask     # search between phi_min1 and phi_max1
    else:
        phi_mask = np.ma.masked_outside(phi, phi_min1, phi_max1, copy=True).mask
    r_mask = np.ma.masked_outside(r, 0, r_radius1, copy=True).mask                  # search inside defined radius
    search_mask = np.ma.mask_or(np.ma.mask_or(phi_mask, r_mask), rms_mask)          # create the search mask

    # Find the co-ordinate of the largest r value in the search area - the first edge point
    r_search = np.ma.masked_array(r, mask = search_mask)
    edge_coord1_yx = np.unravel_index(np.argmax(r_search, axis=None), r_search.shape)
    edge_coord1 = np.array([edge_coord1_yx[1] + 0.5,edge_coord1_yx[0] + 0.5])
    if edge_coord1_yx[0] == 0: invalid_edge_point = True

    # Check that we are moving forwards. Compare the angles of old and new edge points about
    # the previous edge point on the opposite edge.
    if not invalid_edge_point:
        r2, phi2 = PolarCoordinates(flux_array, np.array([prev_edge_pix[2], prev_edge_pix[3]]))
        delta_phi = PiRange(phi2[edge_coord1_yx[0], edge_coord1_yx[1]] - phi2[prev_edge_pix[1], prev_edge_pix[0]])
        moving_backwards = delta_phi > 0

    if not moving_backwards and not invalid_edge_point:
        # Create a mask to search for nearest edge point on the other side of the ridge point
        min_phi = min(phi_min2, phi_max2); max_phi = max(phi_min2, phi_max2)
        quad_min = CheckQuadrant(min_phi); quad_max = CheckQuadrant(max_phi)
        diff = max_phi - min_phi
        if 3 <= quad_min <= 4 and 1 <= quad_max <= 2 and diff > pi:
            phi_mask = np.ma.masked_inside(phi, phi_min2, phi_max2, copy=True).mask     # search between phi_min2 and phi_max2
        else:
            phi_mask = np.ma.masked_outside(phi, phi_min2, phi_max2, copy=True).mask
        r_mask = np.ma.masked_outside(r, 0, r_radius2, copy=True).mask                  # search inside defined radius
        search_mask = np.ma.mask_or(np.ma.mask_or(phi_mask, r_mask), rms_mask)          # create the search mask

        # Find the co-ordinate of the largest r value in the search area - the second edge point
        r_search = np.ma.masked_array(r, mask = search_mask)
        edge_coord2_yx = np.unravel_index(np.argmax(r_search, axis=None), r_search.shape)
        edge_coord2 = np.array([edge_coord2_yx[1] + 0.5,edge_coord2_yx[0] + 0.5])
        if edge_coord2_yx[0] == 0: invalid_edge_point = True

        # Check that we are moving forwards. Compare the angles of old and new edge points about
        # the previous edge point on the opposite edge.
        if not invalid_edge_point:
            r1, phi1 = PolarCoordinates(flux_array, np.array([prev_edge_pix[0], prev_edge_pix[1]]))
            delta_phi = PiRange(phi1[edge_coord2_yx[0], edge_coord2_yx[1]] - phi1[prev_edge_pix[3], prev_edge_pix[2]])
            moving_backwards = delta_phi < 0

        if not moving_backwards and not invalid_edge_point:
            edge_points = np.array([edge_coord1[0], edge_coord1[1], edge_coord2[0], edge_coord2[1]])

    return edge_points

#############################################

def FindInitEdgePoints(flux_array, ridge_point, ridge_phi, ridge_R):

    """
    Returns edge points, on either side of the jet, corresponding to
    the closest distance to that edge from the initial ridge point.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    ridge_point - 1D array
                  ridge point co-ordinates along the ridge line

    ridge_phi - float,
                angular direction of the ridge line

    ridge_R - float,
              distance from the source along the jet in pixels
    
    Constants
    ---------

    Returns
    -----------
    edge_points - 1D array
                  Points on either side of the jet, corresponding to the 
                  closest distance to that edge from the initial ridge point,
                  and their distance from source.
                  (x1,y1,x2,y2,R,ridge_x,ridge_y)

    Notes
    -----------
    """

    # Initialise edge_points array
    edge_points = np.full((1,7), np.nan)

    # Get polar coordinates for flux_array, around the ridge point
    r, phi = PolarCoordinates(flux_array, ridge_point)

    # Fill invalid (nan) values with zeroes in flux_array
    flux_array_valid = np.ma.filled(np.ma.masked_invalid(flux_array), 0)

    # Mask the jet - where flux values are above (JMC.nSig * rms)
    jet_mask = np.ma.masked_where(flux_array_valid > (JMC.nSig * JMS.bgRMS), flux_array_valid).mask

    # Search for an edge at right angles to the ridge direction
    search_phi_range1 = PiRange(ridge_phi + (pi*90/180) - (pi*5/180))     # +/- 5 degrees
    search_phi_range2 = PiRange(ridge_phi + (pi*90/180) + (pi*5/180))
    min_phi = min(search_phi_range1, search_phi_range2); max_phi = max(search_phi_range1, search_phi_range2)
    quad_min = CheckQuadrant(min_phi); quad_max = CheckQuadrant(max_phi)
    diff = max_phi - min_phi
    if 3 <= quad_min <= 4 and 1 <= quad_max <= 2 and diff > pi:
        phi_range_mask = np.ma.masked_inside(phi, search_phi_range2, search_phi_range1).mask
    else:
        phi_range_mask = np.ma.masked_outside(phi, search_phi_range1, search_phi_range2).mask
    search_mask = np.ma.mask_or(phi_range_mask, jet_mask)               # search outside the jet

    # Find the co-ordinate of the smallest r value in the search area - the first edge point 
    r_search = np.ma.masked_array(r, mask = search_mask, copy = True)
    edge_coord1_yx = np.unravel_index(np.argmin(r_search, axis=None), r_search.shape)
    edge_coord1 = np.array([edge_coord1_yx[1] + 0.5,edge_coord1_yx[0] + 0.5])

    # Assume the phi of the opposite edge is at phi for the first edge point + pi radians.
    phi_edge2 = PiRange(phi[edge_coord1_yx] + pi)

    # Mask a small phi range around this.
    search_phi_range1 = PiRange(phi[edge_coord1_yx] + pi - (pi*5/180))  # +/- 5 degrees
    search_phi_range2 = PiRange(phi[edge_coord1_yx] + pi + (pi*5/180))
    min_phi = min(search_phi_range1, search_phi_range2); max_phi = max(search_phi_range1, search_phi_range2)
    quad_min = CheckQuadrant(min_phi); quad_max = CheckQuadrant(max_phi)
    diff = max_phi - min_phi
    if 3 <= quad_min <= 4 and 1 <= quad_max <= 2 and diff > pi:
        phi_range_mask = np.ma.masked_inside(phi, search_phi_range2, search_phi_range1).mask
    else:
        phi_range_mask = np.ma.masked_outside(phi, search_phi_range1, search_phi_range2).mask
    phi_nearest_mask = np.ma.mask_or(phi_range_mask, jet_mask)          # search outside the jet

    # Find the phi value closest to phi_edge2. 
    phi_nearest = np.ma.masked_array(phi, mask = phi_nearest_mask, copy = True)
    nearest_phi_edge2 = phi_nearest.flat[np.abs(phi_nearest - phi_edge2).argmin()]

    # Find the co-ordinate of the smallest r value at this phi - the second edge point 
    search_mask = np.ma.masked_where(np.abs(phi_nearest - nearest_phi_edge2) < 0.001, phi_nearest).mask     # precision of phi = 3 dp's
    r_search = np.ma.masked_array(r, mask = search_mask, copy=True)
    edge_coord2_yx = np.unravel_index(np.argmin(r_search, axis=None), r_search.shape)
    edge_coord2 = np.array([edge_coord2_yx[1] + 0.5,edge_coord2_yx[0] + 0.5])

    # Add to the edge points array, putting first the edge point on the left hand side of the jet,
    # relative to the ridge phi direction.
    edge_points = np.array([edge_coord1[0], edge_coord1[1], edge_coord2[0], edge_coord2[1], ridge_R, ridge_point[0], ridge_point[1]])

    return edge_points

#############################################

def FindSearchAngleRanges(phi_prev1, phi_prev2, ridge_phi):

    """
    Returns the search ranges for finding the next edgepoints

    Parameters
    -----------
    phi_prev1 - float,
                 direction of the previous edge point 1

    phi_prev2 - float,
                angular direction of the previous edge point 2

    ridge_phi - float,
                angular direction of the ridgeline
    
    Constants
    ---------

    Returns
    -----------
    phi_min1 - float,
               minimum in angular search range 1

    phi_max1 - float,
               maximum in angular search range 1

    phi_min2 - float,
               minimum in angular search range 2

    phi_max2 - float,
               maximum in angular search range 2

    Notes
    -----------
    """

    # Find the angle at +90 degs to the ridge phi
    ridge_phi_plus_90 = PiRange(ridge_phi + pi/2)

    # Find which edge point this is closest to in angle, to search for the next edge point on that side of the jet
    min_phi = min(phi_prev1, ridge_phi_plus_90); max_phi = max(phi_prev1, ridge_phi_plus_90)
    quad_min = CheckQuadrant(min_phi); quad_max = CheckQuadrant(max_phi)
    diff_1 = max_phi - min_phi
    if 3 <= quad_min <= 4 and 1 <= quad_max <= 2 and diff_1 > pi: diff_1 = np.abs(diff_1 - 2*pi)
    min_phi = min(phi_prev2, ridge_phi_plus_90); max_phi = max(phi_prev2, ridge_phi_plus_90)
    quad_min = CheckQuadrant(min_phi); quad_max = CheckQuadrant(max_phi)
    diff_2 = max_phi - min_phi
    if 3 <= quad_min <= 4 and 1 <= quad_max <= 2 and diff_2 > pi: diff_2 = np.abs(diff_2 - 2*pi)
    if diff_1 < diff_2:
        search_phi1 = ridge_phi_plus_90             # search at +90 degs to the ridge phi
        search_phi2 = PiRange(ridge_phi - pi/2)     # search at -90 degs to the ridge phi
    else:
        search_phi1 = PiRange(ridge_phi - pi/2)     # search at -90 degs to the ridge phi
        search_phi2 = ridge_phi_plus_90             # search at +90 degs to the ridge phi

    # Define the angular search range on both sides
    semi_search_range = (pi/180) * (JMC.search_angle/2.0)
    phi_min1 = PiRange(search_phi1 - semi_search_range)
    phi_max1 = PiRange(search_phi1 + semi_search_range)
    phi_min2 = PiRange(search_phi2 - semi_search_range)
    phi_max2 = PiRange(search_phi2 + semi_search_range)

    return phi_min1, phi_max1, phi_min2, phi_max2

#############################################

def AddEdgePoints(flux_array, edge_points):

    """
    Interpolates extra edge points at points in the jet where
    a long edge line cuts off parts of the jet flux.

    Parameters
    -----------
    flux_array - 2D array, shape(n,2)
                 raw image array

    edge_points - 2D array, shape(n,7)
                  Points on one arm of the jet, corresponding to the 
                  closest distance to that edge from the corresponding ridge point
                  and their distance from source
    
    Constants
    ---------

    Returns
    -----------
    edge_points_updated - 2D array, shape(n,7)
                          Array with extra interpolated edge points
                          and their distance from source

    Notes
    -----------
    """

    # Initialise edgepoints array
    edge_points_updated = edge_points
    lastpts = np.empty((0,7))

    # Fill invalid (nan) values with zeroes in flux_array
    flux_array = np.ma.filled(np.ma.masked_invalid(flux_array), 0)

    # Mask the flux values below the RMS value (JMC.nSig * rms)
    rms_mask = np.ma.masked_where(flux_array <= (JMC.nSig * JMS.bgRMS), flux_array).mask

    # Loop for each side of the jet
    jet_side = 1
    while jet_side <= 2:

        # Loop down this side and look for long edgelines, which will be likely to cut-off flux
        point_count = 0
        for [x_side1, y_side1, x_side2, y_side2, ridge_R, ridge_x, ridge_y] in edge_points_updated:
            point_count += 1

            if point_count > 1:
                if (ridge_R - lastpts[4]) < (JMC.MaxRFactor * JMC.R_es):   # Test whether the last step size has increased by too much

                    if jet_side == 1:
                        dist = np.sqrt( (x_side1 - lastpts[0])**2 + (y_side1 - lastpts[1])**2 ) # distance between last 2 points on this side
                    else:
                        dist = np.sqrt( (x_side2 - lastpts[2])**2 + (y_side2 - lastpts[3])**2 ) # distance between last 2 points on this side

                    R_diff = ridge_R - lastpts[4]   # change in R since last edgepoint

                    if dist > (JMC.MinIntpolFactor * JMC.R_es):
                        # Distance is long enough to have cut off some flux.
                        # Work out the number of sections to divide the length between last 2 points.
                        num_sections = max( min(np.floor(dist/JMC.R_es + 1).astype('int'), JMC.MaxIntpolSections), 2)

                        R_per_section = R_diff / num_sections                           # change in R per section

                        # Interpolate the new edgepoint for each section
                        section_x1_length = (x_side1 - lastpts[0]) / num_sections       # x section length on one side of the jet
                        section_y1_length = (y_side1 - lastpts[1]) / num_sections       # y section length on one side of the jet
                        section_x2_length = (x_side2 - lastpts[2]) / num_sections       # x section length on other side of the jet
                        section_y2_length = (y_side2 - lastpts[3]) / num_sections       # y section length on other side of the jet

                        last_sect_x1 = lastpts[0]; last_sect_y1 = lastpts[1]
                        last_sect_x2 = lastpts[2]; last_sect_y2 = lastpts[3]
                        sect = 1
                        while sect <= (num_sections-1):
                            R_section = lastpts[4] + (R_per_section * sect)             # R for this section

                            x1 = last_sect_x1 + section_x1_length                       # x section co-ord on one side of the jet
                            y1 = last_sect_y1 + section_y1_length                       # y section co-ord on one side of the jet
                            x2 = last_sect_x2 + section_x2_length                       # x section co-ord on other side of the jet
                            y2 = last_sect_y2 + section_y2_length                       # y section co-ord on other side of the jet

                            # Calculate the next search radius for both sides
                            last_edgepoints = edge_points_side[-1]
                            r_radius1 = np.sqrt( (last_edgepoints[5] - last_edgepoints[0])**2 + (last_edgepoints[6] - last_edgepoints[1])**2 ) * JMC.SearchRadiusIncFactor
                            r_radius2 = np.sqrt( (last_edgepoints[5] - last_edgepoints[2])**2 + (last_edgepoints[6] - last_edgepoints[3])**2 ) * JMC.SearchRadiusIncFactor

                            # Calculate the interpolated ridge point and take polar co-ordinates around it
                            ridge_section_x = lastpts[5] + ((ridge_x - lastpts[5]) / num_sections * sect)
                            ridge_section_y = lastpts[6] + ((ridge_y - lastpts[6]) / num_sections * sect)
                            ridge_point = np.array([ridge_section_x, ridge_section_y])
                            r, phi = PolarCoordinates(flux_array, ridge_point)

                            # Find the next additional edge points
                            if jet_side == 1:
                                edgepoint_yx = FindAdditionalEdgePoint(r, phi, rms_mask, x1, y1, section_x1_length, section_y1_length, r_radius1)
                                if edgepoint_yx[0] > 0:                                 # check edgepoint is valid
                                    edgepoint_side1 = np.add(edgepoint_yx, 0.5)
                                    edgepoint_side2 = np.array([y2,x2])                 # initialise the side2 edgepoint as the interpolated point

                                    # If the distance between points on the opposite jet side is long enough, search for new edge point on this side
                                    dist = np.sqrt( (x_side2 - lastpts[2])**2 + (y_side2 - lastpts[3])**2 )
                                    if dist > (JMC.MinIntpolFactor * JMC.R_es):
                                        edgepoint_yx = FindAdditionalEdgePoint(r, phi, rms_mask, x2, y2, section_x2_length, section_y2_length, r_radius2)
                                        if edgepoint_yx[0] > 0:                         # check edgepoint is valid
                                            edgepoint_side2 = np.add(edgepoint_yx, 0.5) # new edge point has been found

                                    # Check that this edgeline doesn't overlap with the previous edge line
                                    if not CheckForIntersectingEdgeLines(edge_points_side[-1], edgepoint_side1[1] ,edgepoint_side1[0], edgepoint_side2[1], edgepoint_side2[0]):
                                        added_edgepoint_coords = np.array([edgepoint_side1[1] ,edgepoint_side1[0], edgepoint_side2[1], edgepoint_side2[0], R_section, ridge_section_x, ridge_section_y])
                                        edge_points_side = np.vstack((edge_points_side, added_edgepoint_coords))
                            else:
                                edgepoint_yx = FindAdditionalEdgePoint(r, phi, rms_mask, x2, y2, section_x2_length, section_y2_length, r_radius2)
                                if edgepoint_yx[0] > 0:                                 # check edgepoint is valid
                                    edgepoint_side2 = np.add(edgepoint_yx, 0.5)
                                    edgepoint_side1 = np.array([y1,x1])                 # initialise the side1 edgepoint as the interpolated point

                                    # If the distance between points on the opposite jet side is long enough, search for new edge point on this side
                                    dist = np.sqrt( (x_side1 - lastpts[0])**2 + (y_side1 - lastpts[1])**2 )
                                    if dist > (JMC.MinIntpolFactor * JMC.R_es):
                                        edgepoint_yx = FindAdditionalEdgePoint(r, phi, rms_mask, x1, y1, section_x1_length, section_y1_length, r_radius1)
                                        if edgepoint_yx[0] > 0:                         # check edgepoint is valid
                                            edgepoint_side1 = np.add(edgepoint_yx, 0.5) # new edge point has been found

                                    # Check that this edgeline doesn't overlap with the previous edge line
                                    if not CheckForIntersectingEdgeLines(edge_points_side[-1], edgepoint_side1[1] ,edgepoint_side1[0], edgepoint_side2[1], edgepoint_side2[0]):
                                        added_edgepoint_coords = np.array([edgepoint_side1[1] ,edgepoint_side1[0], edgepoint_side2[1], edgepoint_side2[0], R_section, ridge_section_x, ridge_section_y])
                                        edge_points_side = np.vstack((edge_points_side, added_edgepoint_coords))

                            last_sect_x1 = last_sect_x1 + section_x1_length; last_sect_y1 = last_sect_y1 + section_y1_length
                            last_sect_x2 = last_sect_x2 + section_x2_length; last_sect_y2 = last_sect_y2 + section_y2_length
                            sect += 1

                edge_points_side = np.vstack((edge_points_side, np.array([x_side1, y_side1, x_side2, y_side2, ridge_R, ridge_x, ridge_y])))
            else:
                edge_points_side = np.empty((0,7))
                edge_points_side = np.vstack((edge_points_side, np.array([x_side1, y_side1, x_side2, y_side2, ridge_R, ridge_x, ridge_y])))

            lastpts = np.array([x_side1, y_side1, x_side2, y_side2, ridge_R, ridge_x, ridge_y])

        edge_points_updated = edge_points_side
        jet_side += 1

    return edge_points_updated

#############################################

def FindAdditionalEdgePoint(r, phi, rms_mask, x, y, section_x_length, section_y_length, r_radius):

    """
    Find the co-ordinates of an additional edge point

    Parameters
    -----------
    r - 2D array, shape(n,2)
        r polar co-ordinates

    phi - 2D array, shape(n,2)
          phi polar co-ordinates

    rms_mask - 2D array, shape(n,2)
               mask for pixels with flux under rms value

    x, y - float,
           x/y coordinates of section edge point

    section_x/y_length - float,
                         x/y lengths of section edge line

    r_radius - float,
               search radius
    
    Constants
    ---------

    Returns
    -----------
    new_edge_point - 1D array, shape(2)
                     y/x co-ordinates of new edge point

    Notes
    -----------
    """

    # Set up the start and end of the phi range around the section point
    x_start = np.floor(x - section_x_length/2).astype('int'); y_start = np.floor(y - section_y_length/2).astype('int')
    x_end = np.floor(x + section_x_length/2).astype('int'); y_end = np.floor(y + section_y_length/2).astype('int')
    phi_start = phi[y_start,x_start]
    phi_end = phi[y_end,x_end]

    min_phi = min(phi_start, phi_end); max_phi = max(phi_start, phi_end)
    quad_min = CheckQuadrant(min_phi); quad_max = CheckQuadrant(max_phi)
    diff = max_phi - min_phi
    if 3 <= quad_min <= 4 and 1 <= quad_max <= 2 and diff > pi:
        phi_mask = np.ma.masked_inside(phi, phi_start, phi_end, copy = True).mask               # search between phi_start and phi_end
    else:
        phi_mask = np.ma.masked_outside(phi, phi_start, phi_end, copy = True).mask
    r_mask = np.ma.masked_outside(r, 0, r_radius, copy=True).mask # search inside defined radius
    search_mask = np.ma.mask_or(np.ma.mask_or(phi_mask, r_mask), rms_mask)                      # create the search mask

    # Find the co-ordinate of the largest r value in the search area - the new edge point
    r_search = np.ma.masked_array(r, mask = search_mask)
    new_edge_point = np.unravel_index(np.argmax(r_search, axis=None), r_search.shape)

    return new_edge_point

#############################################

def GetJetSections(flux_array, edge_points1, edge_points2, ridge1, ridge2):

    """
    Get parameters for each section of the jet.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    edge_points1 - 2D array, shape(n,7)
                   Points on one arm of the jet, corresponding to the 
                   closest distance to that edge from the corresponding ridge point,
                   and their distance from source.

    edge_points2 - 2D array, shape(n,7)
                   Points on other arm of the jet, corresponding to the 
                   closest distance to that edge from the corresponding ridge point,
                   and their distance from source.

    ridge1 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for one arm of the jet
    
    ridge2 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for other arm of the jet
    
    Constants
    ---------

    Returns
    -----------
    section_params_merged1 - 2D array, shape(n,13)
                             Array with section points (x,y * 4), distance from source
                             and computed parameters for one arm of the jet

    section_params_merged2 - 2D array, shape(n,13)
                             Array with section points (x,y * 4), distance from source
                             and computed parameters for other arm of the jet

    section_perimeters1 - 2D array, shape(n,pSize)
                          Array of perimeter co-ordinates for each merged section (-1 filled)
                          for one arm of the jet

    section_perimeters2 - 2D array, shape(n,pSize)
                          Array of perimeter co-ordinates for each merged section (-1 filled)
                          for other arm of the jet

    Notes
    -----------
    """

    # Fill invalid (nan) values with zeroes in flux_array
    flux_array_valid = np.ma.filled(np.ma.masked_invalid(flux_array), 0)

    # Get the section polygon points
    print('Creating jet sections')
    polygon_points1 = GetSectionPolygons(edge_points1)
    polygon_points2 = GetSectionPolygons(edge_points2)

    # Get flux and volume for each arm of the jet
    print('Calculating flux and volume for sections')
    section_parameters1 = GetSectionParameters(flux_array_valid, polygon_points1, initial_polygon_points = polygon_points2[0,0:8])
    section_parameters2 = GetSectionParameters(flux_array_valid, polygon_points2, initial_polygon_points = polygon_points1[0,0:8])

    # Check for overlapping sections before merging
    CheckForOverlappingSections(1, section_parameters1); CheckForOverlappingSections(2, section_parameters2)

    # Merge sections within distance steps along the jet. Create regions for merged sections.
    print('Merging sections and creating section DS9 regions')
    section_params_merged1, section_perimeters1 = MergeSections(section_parameters1)
    section_params_merged2, section_perimeters2 = MergeSections(section_parameters2)

    return section_params_merged1, section_params_merged2, section_perimeters1, section_perimeters2

#############################################

def Calculate_R(section_parameters):

    """
    Calculate the distances of the section centres along the jet

    Parameters
    -----------
    section_parameters - 2D array, shape(n,13)
                         Array with section points (x,y * 4), distance from source
                         and computed parameters for this arm of the jet
    
    Constants
    ---------

    Returns
    -----------
    updated_section_parameters - 2D array, shape(n,12)
                                 Array with section points (x,y * 4), distance from source
                                 and computed parameters for this arm of the jet

    Notes
    -----------
    """

    # Initialise updated section parameters array
    updated_section_parameters = np.empty((0,12))

    for [x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, flux_section, volume_section, area_section] in section_parameters:

        # Calculate the distance of the section centre
        R_section = (R_section_start + R_section_end) / 2.0

        updated_section_parameters = np.vstack((updated_section_parameters, \
                    np.array([x1,y1, x2,y2, x3,y3, x4,y4, R_section, flux_section, volume_section, area_section])))

    return updated_section_parameters

#############################################

def CheckForOverlappingSections(arm_number, sections):

    """
    Check for overlapping sections.

    Parameters
    -----------
    arm_number - integer
                 jet arm to which the sections belong

    section_parameters - 2D array, shape(n,13)
                         Array with section polygon points (x,y * 4), distance from
                         the source and computed parameters for one arm of the jet
    """

    overlapping_sections = False    # Initalise flag

    icnt = 0
    for [x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, flux_section, volume_section, area_section] in sections:
        icnt += 1
        if icnt > 1:
            # Check if lines intersect, other than at their end-points
            if DoLinesIntersect(last_x1,last_y1,last_x2,last_y2,x1,y1,x2,y2) \
                and not (last_x1 == x1 and last_y1 == y1) \
                and not (last_x2 == x2 and last_y2 == y2):
                overlapping_sections = True
        last_x1 = x1; last_y1 = y1; last_x2 = x2; last_y2 = y2

    if overlapping_sections:
        print("Overlapping sections in arm " + str(arm_number))

#############################################

def CheckForIntersectingEdgeLines(lastEdgePoints, x1, y1, x2, y2):

    """
    Check for intersecting edge lines

    Parameters
    -----------
    lastEdgePoints - 1D array shape(7),
                     last edgepoints

    x1, y1, x2, y2 - float,
                     x/y co-ordinates of latest edge points
                     
    """

    intersecting_edgelines = False    # Initialise flag

    last_x1, last_y1, last_x2, last_y2, ridge_R, ridge_x, ridge_y = lastEdgePoints[:]

    # Check if lines intersect, other than at their end-points
    if DoLinesIntersect(last_x1,last_y1,last_x2,last_y2,x1,y1,x2,y2) \
        and not (last_x1 == x1 and last_y1 == y1) \
        and not (last_x2 == x2 and last_y2 == y2):
        intersecting_edgelines = True

    return intersecting_edgelines

#############################################

def MergeSections(section_parameters):

    """
    Merge sections to a required number for one arm of the jet.
    Create an perimeter array that can be used to create SAOImageDS9 regions for the jet.

    Parameters
    -----------
    section_parameters - 2D array, shape(n,13)
                         Array with section points (x,y * 4), distance from source
                         and computed parameters for one arm of the jet
    
    Constants
    ---------

    Returns
    -----------
    section_params_merged - 2D array, shape(n,13)
                            Array with merged section points (x,y * 4), distance from
                            source and computed parameters for one arm of the jet

    merged_section_perimeters - 2D array, shape(n,pSize)
                                Array of perimeter co-ordinates for each merged section (-1 filled)

    Notes
    -----------
    """
    # Set perimeter array size
    pSize = JMC.max_vertices

    # Initialise perimeter array
    perimeter_array = np.empty((0,pSize))

    # Initialise merged section parameters array
    section_params_merged = np.empty((0,13))

    # Initialise merged section perimeter array
    merged_section_perimeters = np.empty((0,pSize))

    # Initialise minimum change in merge R (> beamsize)
    min_delta_R = JMC.MergeMinDeltaRFactor * np.ceil(JMS.max_beamsize)

    max_R_count = 1; max_R = ( min_delta_R + (JMC.MergeRIncreaseStep ** max_R_count) ) * JMS.equiv_R_factor     # Initialise maximum merge distance                              
    last_section_perimeter = []; pInsertPos = 0                                                                 # Initialise last merged section perimeter list
    last_merged_sections = np.full((1,13), np.nan)                                                              # Initialise last merged sections array
    # Initialise last merged flux/volume/area values
    last_merged_flux_section = 0.0; last_merged_volume_section = 0.0; last_merged_area_section = 0.0

    # Loop around all sections and merge while R is less than the maximum merge R value
    sect_count = 0; last_R_section_end = 0.0
    while (sect_count + 1) <= np.size(section_parameters, 0):
        x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, flux_section, volume_section, area_section = section_parameters[sect_count,:]

        # Test whether we have jumped over too large a gap (don't merge across the gap)
        if (R_section_start - last_R_section_end) > (JMC.MaxRFactor * JMC.R_es):
            # We have jumped too large a gap. Add any latest merged sections.
            if not np.isnan(last_merged_sections).any():
                # Add to the merged section parameters array
                section_params_merged = np.vstack((section_params_merged, last_merged_sections))

                # Add to the merged section perimeters array.
                last_section_perimeter = AddLastPerimeterPoints(pInsertPos, last_section_perimeter, \
                                                                last_merged_sections[4],last_merged_sections[5], last_merged_sections[6],last_merged_sections[7])
                perimeter_array = np.array(last_section_perimeter)
                merged_section_perimeters = np.vstack(( merged_section_perimeters, np.hstack((perimeter_array, np.full((pSize-len(perimeter_array)), np.nan))) ))

                last_merged_sections = np.full((1,13), np.nan)                                                      # Reset last merged sections array
                last_merged_flux_section = 0.0; last_merged_volume_section = 0.0; last_merged_area_section = 0.0    # Reset last merged flux/volume/area values

            # Increase the maximum R until it is greater than R_section_start
            while max_R <= R_section_start:
                max_R_count +=1; max_R += ( min_delta_R + (JMC.MergeRIncreaseStep ** max_R_count) ) * JMS.equiv_R_factor

        # Test whether the maximum merge distance R has been achieved
        if R_section_end > max_R:

            # Split this section at the max_R position. This section now ends at exactly max R. 
            section_parameters = SplitSectionAtR(max_R, sect_count, section_parameters)
            x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, flux_section, volume_section, area_section = section_parameters[sect_count,:]

            # Add this section to the last merged sections and perimeters arrays
            merged_flux = last_merged_flux_section + flux_section               # Add on the flux in the latest section
            merged_volume = last_merged_volume_section + volume_section         # Add on the volume in the latest section
            merged_area = last_merged_area_section + area_section               # Add on the area in the latest section

            if np.isnan(last_merged_sections).any():
                # Initialise the last merged section perimeter list with this section
                pInsertPos, last_section_perimeter = AddPerimeterPoints(pInsertPos, last_section_perimeter, x1,y1, x2,y2, np.full((1,13), np.nan))

                # First section in the last merged sections array
                last_merged_sections = np.array([x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, merged_flux, merged_volume, merged_area])

            else:
                # Add to the merged section perimeter list
                pInsertPos, last_section_perimeter = AddPerimeterPoints(pInsertPos, last_section_perimeter, x1,y1, x2,y2, last_merged_sections)

                # Add to the last merged sections array
                last_merged_sections = np.array([last_merged_sections[0],last_merged_sections[1], last_merged_sections[2],last_merged_sections[3], \
                                                    x3,y3, x4,y4, last_merged_sections[8], R_section_end, merged_flux, merged_volume, merged_area])

            # Add the last merged section to the merged section parameters array
            section_params_merged = np.vstack((section_params_merged, last_merged_sections))

            # Add the last merged section to the merged section perimeters array.
            last_section_perimeter = AddLastPerimeterPoints(pInsertPos, last_section_perimeter, \
                                                            last_merged_sections[4],last_merged_sections[5], last_merged_sections[6],last_merged_sections[7])
            perimeter_array = np.array(last_section_perimeter)
            merged_section_perimeters = np.vstack(( merged_section_perimeters, np.hstack((perimeter_array, np.full((pSize-len(perimeter_array)), np.nan))) ))

            # Reset last merged sections array and last merged flux/volume/area values
            last_merged_sections = np.full((1,13), np.nan)
            last_merged_flux_section = 0.0; last_merged_volume_section = 0.0; last_merged_area_section = 0.0

            # Set to the new maximum R
            max_R_count +=1; max_R += ( min_delta_R + (JMC.MergeRIncreaseStep ** max_R_count) ) * JMS.equiv_R_factor

        else:
            # Adding in this section does not exceed the maximum R
            merged_flux = last_merged_flux_section + flux_section               # Add on the flux in the latest section
            merged_volume = last_merged_volume_section + volume_section         # Add on the volume in the latest section
            merged_area = last_merged_area_section + area_section               # Add on the area in the latest section

            if np.isnan(last_merged_sections).any():
                # Initialise the last merged section perimeter list with this section
                pInsertPos, last_section_perimeter = AddPerimeterPoints(pInsertPos, last_section_perimeter, x1,y1, x2,y2, np.full((1,13), np.nan))

                # First section in the last merged sections array
                last_merged_sections = np.array([x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, merged_flux, merged_volume, merged_area])

            else:
                # Add to the merged section perimeter list
                pInsertPos, last_section_perimeter = AddPerimeterPoints(pInsertPos, last_section_perimeter, x1,y1, x2,y2, last_merged_sections)

                # Add to the last merged sections array
                last_merged_sections = np.array([last_merged_sections[0],last_merged_sections[1], last_merged_sections[2],last_merged_sections[3], \
                                                    x3,y3, x4,y4, last_merged_sections[8], R_section_end, merged_flux, merged_volume, merged_area])

            last_merged_flux_section = merged_flux; last_merged_volume_section = merged_volume; last_merged_area_section = merged_area  # Update last merged flux/volume/area values

        last_R_section_end = R_section_end

        # Last section. Add to the merged section parameters array and the perimeters array if necessary.
        if (sect_count + 1 == np.size(section_parameters, 0)) and not np.isnan(last_merged_sections).any():
            section_params_merged = np.vstack((section_params_merged, last_merged_sections))

            last_section_perimeter = AddLastPerimeterPoints(pInsertPos, last_section_perimeter, \
                                                            last_merged_sections[4],last_merged_sections[5], last_merged_sections[6],last_merged_sections[7])
            perimeter_array = np.array(last_section_perimeter)
            merged_section_perimeters = np.vstack(( merged_section_perimeters, np.hstack((perimeter_array, np.full((pSize-len(perimeter_array)), np.nan))) ))

        sect_count += 1

    return section_params_merged, merged_section_perimeters

#############################################

def SplitSectionAtR(split_R, sect_count, section_parameters):

    """
    Split a section into two, around the splitR distance along the jet.
    Share flux and volume proportionally with area.

    Parameters
    -----------
    split_R - Distance along the jet at which to split the section

    sect_count - index for the current section in section_parameters

    section_parameters - 2D array, shape(n,13)
                         Array with section points (x,y * 4), distance from source
                         and computed parameters for one arm of the jet
    
    Constants
    ---------

    Returns
    -----------
    section_parameters - 2D array, shape(n,13)
                         Input section_parameters with inserted remainder
                         (second part) of split section
    Notes
    -----------
    """
    # Get the values for this section
    x1,y1, x2,y2, x3,y3, x4,y4, \
        R_section_start, R_section_end, flux_section, volume_section, area_section = section_parameters[sect_count,:]

    # Calculate the coordinates at the split R.
    # Do this by finding the points at the relevant fraction of the side-lines. 
    # This looks like it will split the area more accurately than, for example, finding where the line
    # that is perpendicular to the ridge-line, at the split_R point, intersects the side-lines.
    fraction_R = (split_R - R_section_start) / (R_section_end - R_section_start)
    split1_x1 = x1; split1_y1 = y1
    split1_x2 = x2; split1_y2 = y2
    split2_x3 = x3; split2_y3 = y3
    split2_x4 = x4; split2_y4 = y4
    if x4 == -1:
        # 3-point section - look for which side forms the base of the next section
        if (sect_count + 1 == np.size(section_parameters, 0)):
            # Last section - take the 3rd point to be the last point
            split1_x3 = x2 + ((x3 - x2) * fraction_R); split1_y3 = y2 + ((y3 - y2) * fraction_R)
            split1_x4 = x1 + ((x4 - x1) * fraction_R); split1_y4 = y1 + ((y4 - y1) * fraction_R)
            split2_x1 = split1_x4; split2_y1 = split1_y4
            split2_x2 = split1_x3; split2_y2 = split1_y3
        else:
            next_section = section_parameters[(sect_count+1),:]
            next_x1 = next_section[0]; next_y1 = next_section[1]; next_x2 = next_section[2]; next_y2 = next_section[3]
            if x1 == next_x1 and y1 == next_y1:
                split1_x3 = x2 + ((x3 - x2) * fraction_R); split1_y3 = y2 + ((y3 - y2) * fraction_R)
                split1_x4 = -1; split1_y4 = -1
                split2_x1 = split1_x1; split2_y1 = split1_y1
                split2_x2 = split1_x3; split2_y2 = split1_y3
            elif x2 == next_x2 and y2 == next_y2:
                split1_x3 = x1 + ((x3 - x1) * fraction_R); split1_y3 = y1 + ((y3 - y1) * fraction_R)
                split1_x4 = -1; split1_y4 = -1
                split2_x1 = split1_x3; split2_y1 = split1_y3
                split2_x2 = split1_x2; split2_y2 = split1_y2
            else:
                # We have jumped a gap - take the 3rd point to be the last point
                split1_x3 = x2 + ((x3 - x2) * fraction_R); split1_y3 = y2 + ((y3 - y2) * fraction_R)
                split1_x4 = x1 + ((x4 - x1) * fraction_R); split1_y4 = y1 + ((y4 - y1) * fraction_R)
                split2_x1 = split1_x4; split2_y1 = split1_y4
                split2_x2 = split1_x3; split2_y2 = split1_y3
    else:
        # 4-point section
        split1_x3 = x2 + ((x3 - x2) * fraction_R); split1_y3 = y2 + ((y3 - y2) * fraction_R)
        split1_x4 = x1 + ((x4 - x1) * fraction_R); split1_y4 = y1 + ((y4 - y1) * fraction_R)
        split2_x1 = split1_x4; split2_y1 = split1_y4
        split2_x2 = split1_x3; split2_y2 = split1_y3

    # Get the area fraction
    split1_area = GetArea(np.array([split1_x1,split1_y1, split1_x2,split1_y2, split1_x3,split1_y3, split1_x4,split1_y4]))
    total_area = GetArea(np.array([x1,y1, x2,y2, x3,y3, x4,y4]))
    if (abs(split1_area) > abs(total_area)) or total_area == 0:
        # Area of the split is not a fraction of the section area (this can happen for extremely small sections).
        # Set the fraction to zero.
        fraction_area = 0.0
    else:
        fraction_area = split1_area / total_area

    # Get all remaining values for the two splits
    split1_R_start = R_section_start
    split1_R_end = split_R
    split2_R_start = split_R
    split2_R_end = R_section_end

    split1_area = area_section * fraction_area
    split2_area = area_section * (1.0 - fraction_area)

    split1_flux = flux_section * fraction_area
    split2_flux = flux_section * (1.0 - fraction_area)

    split1_volume = GetVolume(np.array([split1_x1,split1_y1, split1_x2,split1_y2, split1_x3,split1_y3, split1_x4,split1_y4]))
    if abs(split1_volume) > abs(volume_section):
        # Volume of the split is not a fraction of the section volume (this can happen for extremely small sections).
        # Approximate the volume fraction as the area fraction.
        split1_volume = volume_section * fraction_area
    split2_volume = volume_section - split1_volume

    # Update the first split and add the second split into the section_parameters array
    split1_values = np.array([split1_x1,split1_y1, split1_x2,split1_y2, split1_x3,split1_y3, split1_x4,split1_y4, \
                              split1_R_start,split1_R_end, split1_flux, split1_volume, split1_area])
    split2_values = np.array([split2_x1,split2_y1, split2_x2,split2_y2, split2_x3,split2_y3, split2_x4,split2_y4, \
                              split2_R_start,split2_R_end, split2_flux, split2_volume, split2_area])
    section_parameters = np.delete(section_parameters, sect_count, axis=0)
    section_parameters = np.insert(section_parameters, sect_count, split1_values, axis=0)
    section_parameters = np.insert(section_parameters, sect_count+1, split2_values, axis=0)

    return section_parameters

#############################################

def AddPerimeterPoints(pInsertPos, section_perimeter, x1,y1, x2,y2, last_merged_sections):

    """
    Add the perimeter points for a section

    Parameters
    -----------
    pInsertPos - Integer
                 Position in the section perimeter list at which to insert the points

    section_perimeter - list
                        Current list of perimeter points for a section

    x1,y1, x2,y2 - float
                   x/y co-ordinates of the vertices of the section

    last_merged_sections - 1D array
                           Co-ordinates and other data for latest merged section

    Returns
    -----------
    pInsertPos - Integer
                 Updated position in the section perimeter list at which to insert next points

    section_perimeter - list
                        Updated list of perimeter points for a section
    """

    if np.isnan(last_merged_sections).any():                        # New set of perimeter points
        section_perimeter = [x1]; 
        section_perimeter.append(y1); section_perimeter.append(x2); section_perimeter.append(y2); pInsertPos = 4

    else:                                                           # Add base points of section (if not already added for 3 vertex section)
        if (x1 == last_merged_sections[0] and y1 == last_merged_sections[1]):
            section_perimeter.insert(pInsertPos, y2); section_perimeter.insert(pInsertPos, x2); pInsertPos += 2

        elif (x2 == last_merged_sections[2] and y2 == last_merged_sections[3]):
            section_perimeter.insert(pInsertPos, y1); section_perimeter.insert(pInsertPos, x1)
        else:
            section_perimeter.insert(pInsertPos, y2); section_perimeter.insert(pInsertPos, x2); pInsertPos += 2
            section_perimeter.insert(pInsertPos, y1); section_perimeter.insert(pInsertPos, x1)

    return pInsertPos, section_perimeter

#############################################

def AddLastPerimeterPoints(pInsertPos, section_perimeter, x3,y3, x4,y4):

    """
    Add the perimeter points for a section

    Parameters
    -----------
    pInsertPos - Integer
                 Position in the section perimeter list at which to insert the points

    section_perimeter - list
                        Current list of perimeter points for a section

    x3,y3, x4,y4 - float
                   x/y co-ordinates of the vertices of the section

    Returns
    -----------
    section_perimeter - list
                        Updated list of perimeter points for a section
    """

    if x4 == -1:                            # Section has 3 vertices
        section_perimeter.insert(pInsertPos, y3); section_perimeter.insert(pInsertPos, x3)

    else:                                   # Section has 4 vertices
        section_perimeter.insert(pInsertPos, y4); section_perimeter.insert(pInsertPos, x4)
        section_perimeter.insert(pInsertPos, y3); section_perimeter.insert(pInsertPos, x3)

    return section_perimeter

#############################################

def GetSectionParameters(flux_array, polygon_points, initial_polygon_points):

    """
    Get parameters for each section polygon in the jet e.g. flux, volume.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    polygon_points - 2D array, shape(n,10)
                     Array with section polygon points (x,y * 4) and distance from
                     the source.

    initial_polygon_points - 1D array
                             Array with initial section polygon points (x,y * 4) and 
                             distance from the source.
    
    Constants
    ---------

    Returns
    -----------
    section_parameters - 2D array, shape(n,13)
                         Array with section polygon points (x,y * 4), distance from
                         the source and computed parameters for one arm of the jet


    Notes
    -----------
    """

    # Initialise section parameters array
    section_parameters = np.empty((0,13))

    # Initialise the initial polygon points (used for pixel overlap masking)
    last_polypoints = initial_polygon_points

    # Loop through the section polygons for one arm of the jet
    sect_count = 0
    for [x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end] in polygon_points:
        sect_count += 1

        polypoints = np.array([x1,y1, x2,y2, x3,y3, x4,y4])         # Set the current section polygon points
        if sect_count + 1 > np.size(polygon_points, 0):             # Set the next section polygon points
            next_polypoints = np.full((1,8), np.nan)
        else:
            next_polypoints = polygon_points[sect_count,0:8]

        # Get the section flux and area
        section_flux, section_area = GetFlux(flux_array, polypoints, last_polypoints, next_polypoints)

        # Get the section volume
        section_volume = GetVolume(polypoints)

        # Add section co-ordinates and parameters to the array
        section_parameters = np.vstack((section_parameters, 
                                        np.array([x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, section_flux, section_volume, section_area])))

        last_polypoints = np.array([x1,y1, x2,y2, x3,y3, x4,y4])    # Save the last section polygon points          

    return section_parameters

#############################################

def GetSectionPolygons(edge_points):

    """
    Returns an array of section polygon vertices and distance from the source.

    Parameters
    -----------
    edge_points - 2D array, shape(n,7)
                  Points on one arm of the jet, corresponding to the 
                  closest distance to that edge from the corresponding ridge point.
    
    Constants
    ---------

    Returns
    -----------
    polygon_points - 2D array, shape(n,10)
                     Array with section polygon points (x,y * 4) and distance from
                     the source.


    Notes
    -----------
    """

    # Initialise polygon_points array
    polygon_points = np.empty((0,10))

    # Initialise distance from source
    last_R_section = 0.0

    # Loop through the edge points and create the polygon arrays.
    point_count = 0
    for [x1, y1, x2, y2, R_section, ridge_x, ridge_y] in edge_points:
        point_count += 1

        if point_count > 1:

            if (R_section - last_R_section) < (JMC.MaxRFactor * JMC.R_es): # Test whether the last step size has increased by too much

                # Look at the lengths of the sides of the new polygon
                x1_diff = np.abs(x1 - lastpts[0]); y1_diff = np.abs(y1 - lastpts[1])
                x2_diff = np.abs(x2 - lastpts[2]); y2_diff = np.abs(y2 - lastpts[3])

                # Check the distance between current and previous points
                r1_diff = np.sqrt(x1_diff**2 + y1_diff**2)
                r2_diff = np.sqrt(x2_diff**2 + y2_diff**2)

                # Check that there are no duplicate co-ordinates
                if x1_diff < 0.1 and y1_diff < 0.1 and x2_diff < 0.1 and y2_diff < 0.1:
                    # Ignore duplicate points
                    x1 = lastpts[0]; y1 = lastpts[1]; x2 = lastpts[2]; y2 = lastpts[3]
                else:
                    # Check if lines between previous edge points and between current edge points intersect
                    if DoLinesIntersect(lastpts[0],lastpts[1],lastpts[2],lastpts[3],x1,y1,x2,y2):
                        # Use a 3-vertex polygon - test which point to use as the 3rd vertex
                        if r1_diff < r2_diff:
                            current_polygon_points = np.array([lastpts[0],lastpts[1], lastpts[2],lastpts[3], x2,y2, -1,-1, last_R_section, R_section])
                            x1 = lastpts[0]; y1 = lastpts[1]
                        else:
                            current_polygon_points = np.array([lastpts[0],lastpts[1], lastpts[2],lastpts[3], x1,y1, -1,-1, last_R_section, R_section])
                            x2 = lastpts[2]; y2 = lastpts[3]
                    else:
                        # Lines do not intersect. Determine whether the section has 3 or 4 vertices
                        if (x1_diff < 0.1 and y1_diff < 0.1) or (r1_diff < 0.5):
                            current_polygon_points = np.array([lastpts[0],lastpts[1], lastpts[2],lastpts[3], x2,y2, -1,-1, last_R_section, R_section])
                            x1 = lastpts[0]; y1 = lastpts[1]
                        elif (x2_diff < 0.1 and y2_diff < 0.1) or (r2_diff < 0.5):
                            current_polygon_points = np.array([lastpts[0],lastpts[1], lastpts[2],lastpts[3], x1,y1, -1,-1, last_R_section, R_section])
                            x2 = lastpts[2]; y2 = lastpts[3]
                        else:
                            current_polygon_points = np.array([lastpts[0],lastpts[1], lastpts[2],lastpts[3], x2,y2, x1,y1, last_R_section, R_section])

                    # Add section co-ordinates and parameters to the array
                    polygon_points = np.vstack((polygon_points, current_polygon_points))

                    last_R_section = R_section
            else:
                last_R_section = R_section
                point_count = 1     # We have jumped too large a gap. Reset the count.

        lastpts = np.array([x1, y1, x2, y2])

    return polygon_points

#############################################

def DoLinesIntersect(line1_x1, line1_y1, line1_x2, line1_y2, line2_x1, line2_y1, line2_x2, line2_y2):

    """
    Returns a flag to show if the two supplied lines intersect.

    Parameters
    -----------
    line1_x1, line1_y1 - co-ordinates of one end of line 1

    line1_x2, line1_y2 - co-ordinates of other end of line 1

    line2_x1, line2_y1 - co-ordinates of one end of line 2

    line2_x2, line2_y2 - co-ordinates of other end of line 2
    
    Constants
    ---------

    Returns
    -----------
    lines_intersect - boolean
                      True if lines intersect

    Notes
    -----------
    """

    # Initialise intersect flag
    lines_intersect = False

    # Setup the lines
    line1_point1 = (line1_x1, line1_y1)
    line1_point2 = (line1_x2, line1_y2)
    line2_point1 = (line2_x1, line2_y1)
    line2_point2 = (line2_x2, line2_y2)
    line1 = LineString([line1_point1, line1_point2])
    line2 = LineString([line2_point1, line2_point2])

    # Look for an intersection
    int_pt = line1.intersection(line2)
    lines_intersect = not int_pt.is_empty

    return lines_intersect

#############################################

def LineIntersection(line1_x1, line1_y1, line1_x2, line1_y2, line2_x1, line2_y1, line2_x2, line2_y2):

    """
    Returns the intersection point of the two supplied lines
    (nan if no intersection).

    Parameters
    -----------
    line1_x1, line1_y1 - co-ordinates of one end of line 1

    line1_x2, line1_y2 - co-ordinates of other end of line 1

    line2_x1, line2_y1 - co-ordinates of one end of line 2

    line2_x2, line2_y2 - co-ordinates of other end of line 2
    
    Constants
    ---------

    Returns
    -----------
    intersection_point - 1D array shape(2)
                         intersection co-ordinate (nan if no intersection)

    Notes
    -----------
    """

    # Setup the lines
    line1_point1 = (line1_x1, line1_y1)
    line1_point2 = (line1_x2, line1_y2)
    line2_point1 = (line2_x1, line2_y1)
    line2_point2 = (line2_x2, line2_y2)
    line1 = LineString([line1_point1, line1_point2])
    line2 = LineString([line2_point1, line2_point2])

    # Look for an intersection
    int_pt = line1.intersection(line2)
    if int_pt.is_empty:
        intersection_point = nan
    else:
        intersection_point = np.array([int_pt.x, int_pt.y])

    return intersection_point

#############################################

def GetFlux(flux_array, curr_polypoints, last_polypoints, next_polypoints):

    """
    Returns the total flux in this section, sharing any overlap flux 
    with adjacent sections.

    Parameters
    -----------
    flux_array - 2D array, shape(n,2)
                 raw image array

    curr_polypoints - 1D array
                      Co-ordinates of the section polygon vertices.

    last_polypoints - 1D array
                      Co-ordinates of the last section polygon vertices.

    next_polypoints - 1D array
                      Co-ordinates of the next section polygon vertices.
    
    Constants
    ---------

    Returns
    -----------
    section_flux - float
                   Total flux for this section of the jet

    section_pixel_count = float
                          Total pixels for this section (section area)

    Notes
    -----------
    """

    # Create the current polygon mask
    if curr_polypoints[6] == -1:
        polygon_points = np.array([[curr_polypoints[1],curr_polypoints[0]], [curr_polypoints[3],curr_polypoints[2]], 
                                   [curr_polypoints[5],curr_polypoints[4]]])
    else:
        polygon_points = np.array([[curr_polypoints[1],curr_polypoints[0]], [curr_polypoints[3],curr_polypoints[2]], 
                                   [curr_polypoints[5],curr_polypoints[4]], [curr_polypoints[7],curr_polypoints[6]]])
    curr_polygon_mask = polygon2mask(flux_array.shape, polygon_points)

    # Create the last polygon mask
    if last_polypoints[6] == -1:
        polygon_points = np.array([[last_polypoints[1],last_polypoints[0]], [last_polypoints[3],last_polypoints[2]], 
                                   [last_polypoints[5],last_polypoints[4]]])
    else:
        polygon_points = np.array([[last_polypoints[1],last_polypoints[0]], [last_polypoints[3],last_polypoints[2]], 
                                   [last_polypoints[5],last_polypoints[4]], [last_polypoints[7],last_polypoints[6]]])
    last_polygon_mask = polygon2mask(flux_array.shape, polygon_points)

    # Create the next polygon mask
    if not np.isnan(next_polypoints).any():         # Are we on the last section?
        if next_polypoints[6] == -1:
            polygon_points = np.array([[next_polypoints[1],next_polypoints[0]], [next_polypoints[3],next_polypoints[2]], 
                                       [next_polypoints[5],next_polypoints[4]]])
        else:
            polygon_points = np.array([[next_polypoints[1],next_polypoints[0]], [next_polypoints[3],next_polypoints[2]], 
                                       [next_polypoints[5],next_polypoints[4]], [next_polypoints[7],next_polypoints[6]]])
        next_polygon_mask = polygon2mask(flux_array.shape, polygon_points)
    else:
        next_polygon_mask = np.ma.make_mask(np.zeros_like(flux_array))

    # Make sure there is no overlap with the last and next sections
    curr_flux_mask = np.ma.mask_or(np.ma.mask_or((~curr_polygon_mask), last_polygon_mask), next_polygon_mask)

    # Get overlap masks
    last_overlap_mask = np.logical_and(curr_polygon_mask, last_polygon_mask)
    next_overlap_mask = np.logical_and(curr_polygon_mask, next_polygon_mask)

    # Determine the section and overlap pixel counts
    curr_pixel_count = (~curr_flux_mask).sum()
    last_overlap_pixel_count = last_overlap_mask.sum(); next_overlap_pixel_count = next_overlap_mask.sum()
    overlap_pixel_count = last_overlap_pixel_count + next_overlap_pixel_count
    section_pixel_count = (curr_pixel_count + overlap_pixel_count / 2.)                   # share overlapping pixel count

    # Sum the flux in this section, sharing any overlap flux with adjacent sections
    if curr_pixel_count > 0: 
        flux_curr_polygon = np.ma.masked_array(flux_array, curr_flux_mask, copy = True).sum()
    else: 
        flux_curr_polygon = 0.0
    if last_overlap_pixel_count > 0: 
        flux_last_overlap = np.ma.masked_array(flux_array, (~last_overlap_mask), copy = True).sum()
    else: 
        flux_last_overlap = 0.0
    if next_overlap_pixel_count > 0:
        flux_next_overlap = np.ma.masked_array(flux_array, (~next_overlap_mask), copy = True).sum()
    else:
        flux_next_overlap = 0.0
    section_flux = flux_curr_polygon + ((flux_last_overlap + flux_next_overlap) / 2.)   # share overlapping pixel flux

    # Take account of the background flux, ensuring that resulting flux is >= 0
    background_flux = JMS.bgMean * section_pixel_count
    section_flux = max((section_flux-background_flux), 0.0)

    return section_flux, section_pixel_count

#############################################

def GetArea(polypoints):

    """
    Returns the total area of this section.

    Parameters
    -----------
    polypoints - 1D array
                 Co-ordinates of the section polygon vertices.
    
    Constants
    ---------

    Returns
    -----------
    section_area - float (cubic pixels)
                   Area for this section of the jet

    Notes
    -----------
    """

    if polypoints[6] == -1:
        points = polypoints[0:6]    # 3 points in polygon
    else:
        points = polypoints         # 4 points in polygon

    points_x = points[::2]          # x (even) co-ordinates
    points_y = points[1::2]         # y (odd) coordinates

    return Polygon(zip(points_x, points_y)).area

#############################################

def GetVolume(polypoints):

    """
    Returns the total volume of this section, by assuming the
    3D section is a cone with the vertex sliced off at an angle.
    for a 3-point section, assume the code is cut by a slant plane
    so that the cut just touches the base.

    Parameters
    -----------
    polypoints - 1D array
                 Co-ordinates of the section polygon vertices.
    
    Constants
    ---------

    Returns
    -----------
    section_volume - float (cubic pixels)
                     Volume for this section of the jet

    Notes
    -----------
    """

    if polypoints[6] == -1:

        # 3 points in polygon
        base_points, top_point = Setup3PointPolygon(polypoints)

        # Calculate the volume.
        # Assume the volume is of the base of a cone, sliced by a slant plane down to the X axis.
        if abs(base_points[1,0] - top_point[0]) < JMC.geo_lentol or abs(top_point[1]) < JMC.geo_lentol:   # Test the gradient of the slant plane
            # Gradient of slant line is infinite or zero. Assume the volume is given by a sliced cylinder.
            cylinder_H = top_point[1]                                           # Height of cylinder
            cylinder_R = base_points[1,0] / 2                                   # Radius of cylinder
            section_volume = pi * cylinder_R**2 * cylinder_H / 2
        else:
            cone_R = base_points[1,0] / 2                                           # Radius of cone base.
            cone_cotPhi = top_point[1] / top_point[0]                               # cot of cone half-angle.
            cone_H = cone_R * cone_cotPhi                                           # Height of cone.
            cone_slant_m = np.abs(top_point[1] / (base_points[1,0] - top_point[0])) # Gradient of slant plane.
            cone_tanTheta = cone_slant_m                                            # Angle from horizontal of slant plane.

            # If theta >= (pi/2 - phi), the calculation becomes infinite.
            # In this case, assume the volume is given by a sliced cylinder.
            phi_rdns = atan(1.0/cone_cotPhi)
            theta_rdns = atan(cone_tanTheta)
            if abs(theta_rdns + JMC.geo_angtol) < (pi/2 - abs(phi_rdns)):

                # Volume of top of the cone, from the vertex down to the slant plane
                cone_volume_top = pi/3 * cone_R**2 * cone_H * \
                                  pow( ((cone_H - (cone_tanTheta * cone_R)) / (cone_H + (cone_tanTheta * cone_R))), 3/2)

                # Total volume of the cone
                cone_volume = pi/3 * cone_R**2 * cone_H

                # Volume of the base of the cone, from the base up to the slant plane
                cone_volume_base = cone_volume - cone_volume_top
                section_volume = cone_volume_base

            else:                                                                   # Assume sliced cylinder
                cylinder_H = top_point[1]                                           # Height of cylinder
                cylinder_R = base_points[1,0] / 2                                   # Radius of cylinder
                section_volume = pi * cylinder_R**2 * cylinder_H / 2

    else:
        # 4 points in polygon
        base_points, top_points = Setup4PointPolygon(polypoints)

        # Calculate the volume.
        # Assume the volume is of the base of a cone, sliced by a slant plane.
        if abs(top_points[1,0] - top_points[0,0]) < JMC.geo_lentol or \
           abs(top_points[0,1]) < JMC.geo_lentol or abs(top_points[1,1]) < JMC.geo_lentol:    # Test the gradient of the slant plane and heights of the top points
            # Gradient  of slant line is infinite or zero. Assume the volume is given by a sliced cylinder.
            cylinder_H1 = top_points[0,1]                                           # First height of cylinder
            cylinder_H2 = top_points[1,1]                                           # Second height of cylinder
            cylinder_R = base_points[1,0] / 2                                       # Radius of cylinder
            section_volume = pi * cylinder_R**2 * (cylinder_H1 + cylinder_H2) / 2
        else:
            cone_R = base_points[1,0] / 2                                                                       # Radius of cone base.
            cone_cotPhi = top_points[0,1] / top_points[0,0]                                                     # cot of cone half-angle.
            cone_H = cone_R * cone_cotPhi                                                                       # Height of cone.
            cone_slant_m = np.abs((top_points[1,1] - top_points[0,1]) / (top_points[1,0] - top_points[0,0]))    # Gradient of slant plane.
            cone_tanTheta = cone_slant_m                                                                        # Angle from horizontal of slant plane.
            if top_points[0,1] < top_points[1,1]:                                       # Test for the smallest side
                cone_h1 = ( cone_slant_m * (base_points[1,0] / 2) ) + top_points[0,1]   # Distance of slant plane above the base, along the cone axis.
            else:
                cone_h1 = ( cone_slant_m * (base_points[1,0] / 2) ) + top_points[1,1]   # Distance of slant plane above the base, along the cone axis.
            cone_h = cone_H - cone_h1                                                   # Distance of slant plane from the vertex, along the cone axis.

            # If theta >= (pi/2 - phi), the calculation becomes infinite.
            # In this case, assume the volume is given by a sliced cylinder.
            phi_rdns = atan(1.0/cone_cotPhi)
            theta_rdns = atan(cone_tanTheta)
            if abs(theta_rdns + JMC.geo_angtol) < (pi/2 - abs(phi_rdns)):

                # Volume of top of the cone, from the vertex down to the slant plane
                cone_volume_top = (pi/3 * pow(cone_h,3) * cone_cotPhi) / pow( (cone_cotPhi**2 - cone_tanTheta**2), 3/2)

                # Total volume of the cone
                cone_volume = pi/3 * cone_R**2 * cone_H

                # Volume of the base of the cone, from the base up to the slant plane
                cone_volume_base = cone_volume - cone_volume_top
                section_volume = cone_volume_base
            else:                                                                   # Assume sliced cylinder
                cylinder_H1 = top_points[0,1]                                       # First height of cylinder
                cylinder_H2 = top_points[1,1]                                       # Second height of cylinder
                cylinder_R = base_points[1,0] / 2                                   # Radius of cylinder
                section_volume = pi * cylinder_R**2 * (cylinder_H1 + cylinder_H2) / 2

    return section_volume

#############################################

def Setup3PointPolygon(polypoints):

    """
    Sets up a 3-point section polygon, ready for the volume calculation.

    Parameters
    -----------
    polypoints - 1D array
                 Co-ordinates of the section polygon vertices.
    
    Constants
    ---------

    Returns
    -----------
    base_points - 1D array
                  Co-ordinates of the polygon base

    top_point - 1D array
                Co-ordinate of the polygon top

    Notes
    -----------
    """

    # Determine the base and top co-ordinate of the polygon "partial cone".
    length1 = np.sqrt( (polypoints[2] - polypoints[0])**2 + (polypoints[3] - polypoints[1])**2 )
    length2 = np.sqrt( (polypoints[4] - polypoints[2])**2 + (polypoints[5] - polypoints[3])**2 )
    length3 = np.sqrt( (polypoints[0] - polypoints[4])**2 + (polypoints[1] - polypoints[5])**2 )
    lengths = np.array([length1, length2, length3])
    arg_maxlen = np.argmax(lengths)
    if arg_maxlen == 0:
         base_points = np.array([[polypoints[0], polypoints[1]], [polypoints[2], polypoints[3]]])       # Base co-ordinates
         top_point = np.array([polypoints[4], polypoints[5]])                                           # Top co-ordinate
    elif arg_maxlen == 1:
         base_points = np.array([[polypoints[2], polypoints[3]], [polypoints[4], polypoints[5]]])       # Base co-ordinates
         top_point = np.array([polypoints[0], polypoints[1]])                                           # Top co-ordinate
    else:
         base_points = np.array([[polypoints[4], polypoints[5]], [polypoints[0], polypoints[1]]])       # Base co-ordinates
         top_point = np.array([polypoints[2], polypoints[3]])                                           # Top co-ordinate

    # Translate the polygon so that first base point is at the origin
    offset = np.array([base_points[0,0], base_points[0,1]])
    base_points -= offset; top_point -= offset 

    # Rotate the polygon so that the base is on the positive X axis
    angle_to_x = atan2(base_points[1,1], base_points[1,0])
    rot_angle = 2*pi - TwoPiRange(angle_to_x)
    anchor = np.array([0.0,0.0])
    shape_to_rotate = np.vstack((base_points, top_point))
    rotated_shape = dot( shape_to_rotate-anchor, np.array([[cos(rot_angle),sin(rot_angle)], [-sin(rot_angle),cos(rot_angle)]]) ) + anchor
    top_point = np.array([rotated_shape[2,0],rotated_shape[2,1]])
    base_points = np.array([[rotated_shape[0,0],rotated_shape[0,1]], [rotated_shape[1,0],rotated_shape[1,1]]])
    base_points[0,0] = 0.0; base_points[0,1] = 0.0                  # Ensure first base point is exactly at the origin
    base_points[1,1] = 0.0                                          # Ensure second base point lies exactly on the X axis

    # If the top point has a -ve Y co-ordinate, reflect around the X axis
    if top_point[1] < 0: top_point[1] = - top_point[1]

    # Reflect the top X co-ordinate around the cone axis, such
    # that the left side is steeper than the right side
    base_midpoint = base_points[1,0] / 2
    if top_point[0] > base_midpoint:
        top_point[0] = base_midpoint - (top_point[0] - base_midpoint)

    # If the top point is less than 1 degree from the cone axis, move the   
    # top X co-ordinate such that the top point is 1 degree from the cone axis
    if atan2((base_midpoint - top_point[0]), top_point[1]) < pi*1/180:
        top_point[0] = base_midpoint - (top_point[1] * tan(pi*1/180))

    # if the top point is less than 1 degree from the Y axis, move the   
    # top X co-ordinate such that the top point is 1 degree from the Y axis
    if np.abs(atan2(top_point[0], top_point[1])) < pi*1/180:
        top_point[0] = top_point[1] * tan(pi*1/180)

    return base_points, top_point

#############################################

def Setup4PointPolygon(polypoints):

    """
    Sets up a 4-point section polygon, ready for the volume calculation.

    Parameters
    -----------
    polypoints - 1D array
                 Co-ordinates of the section polygon vertices.
    
    Constants
    ---------

    Returns
    -----------
    base_points - 1D array
                  Co-ordinates of the polygon base

    top_point - 1D array
                Co-ordinates of the polygon top

    Notes
    -----------
    """

    # Determine the base and top co-ordinates of the polygon "partial cone".
    length1 = np.sqrt( (polypoints[2] - polypoints[0])**2 + (polypoints[3] - polypoints[1])**2 )
    length2 = np.sqrt( (polypoints[4] - polypoints[2])**2 + (polypoints[5] - polypoints[3])**2 )
    length3 = np.sqrt( (polypoints[6] - polypoints[4])**2 + (polypoints[7] - polypoints[5])**2 )
    length4 = np.sqrt( (polypoints[0] - polypoints[6])**2 + (polypoints[1] - polypoints[7])**2 )
    lengths = np.array([length1, length2, length3, length4])
    arg_maxlen = np.argmax(lengths)
    if arg_maxlen == 0:
        base_points = np.array([[polypoints[0], polypoints[1]], [polypoints[2], polypoints[3]]])        # Base co-ordinates
        top_points = np.array([[polypoints[4], polypoints[5]], [polypoints[6], polypoints[7]]])         # Top co-ordinates
    elif arg_maxlen == 1:
        base_points = np.array([[polypoints[2], polypoints[3]], [polypoints[4], polypoints[5]]])        # Base co-ordinates
        top_points = np.array([[polypoints[6], polypoints[7]], [polypoints[0], polypoints[1]]])         # Top co-ordinates
    elif arg_maxlen == 2:
        base_points = np.array([[polypoints[4], polypoints[5]], [polypoints[6], polypoints[7]]])        # Base co-ordinates
        top_points = np.array([[polypoints[0], polypoints[1]], [polypoints[2], polypoints[3]]])         # Top co-ordinates
    else:
        base_points = np.array([[polypoints[6], polypoints[7]], [polypoints[0], polypoints[1]]])        # Base co-ordinates
        top_points = np.array([[polypoints[2], polypoints[3]], [polypoints[4], polypoints[5]]])         # Top co-ordinates

    # Translate the polygon so that first base point is at the origin
    offset = np.array([base_points[0,0], base_points[0,1]])
    base_points -= offset; top_points -= offset 

    # Rotate the polygon so that the base is on the positive X axis
    angle_to_x = atan2(base_points[1,1], base_points[1,0])
    rot_angle = 2*pi - TwoPiRange(angle_to_x)
    anchor = np.array([0.0,0.0])
    shape_to_rotate = np.concatenate((base_points, top_points), axis = 0)
    rotated_shape = dot( shape_to_rotate-anchor, np.array([[cos(rot_angle),sin(rot_angle)], [-sin(rot_angle),cos(rot_angle)]]) ) + anchor
    base_points = np.array([[rotated_shape[0,0],rotated_shape[0,1]], [rotated_shape[1,0],rotated_shape[1,1]]])
    top_points = np.array([[rotated_shape[3,0],rotated_shape[3,1]], [rotated_shape[2,0],rotated_shape[2,1]]])
    base_points[0,0] = 0.0; base_points[0,1] = 0.0                  # Ensure first base point is exactly at the origin
    base_points[1,1] = 0.0                                          # Ensure second base point lies exactly on the X axis

    # If the top points have -ve Y co-ordinates, reflect around the X axis.
    # [Note that, after identifying the base points as the longest line, it is rare that one of the
    #  top points is above and one below the X axis. In this case, one will only be slightly below.]
    if top_points[0,1] < 0: top_points[0,1] = - top_points[0,1]
    if top_points[1,1] < 0: top_points[1,1] = - top_points[1,1]

    # If the gradient of the first side is negative or the gradient of the second side is positive,
    # move the X co-ordinate such that the side is vertical.
    if (top_points[0,0] - base_points[0,0]) < 0: top_points[0,0] = base_points[0,0]
    if (top_points[1,0] - base_points[1,0]) > 0: top_points[1,0] = base_points[1,0]

    # If the sides of the polygon are less than 1 degree from the vertical, move the   
    # top co-ordinates to make the sides 1 degree from the vertical.
    if atan2(top_points[0,0], top_points[0,1]) < pi*1/180:
        top_points[0,0] = top_points[0,1] * tan(pi*1/180)
    if atan2((base_points[1,0] - top_points[1,0]), top_points[1,1]) < pi*1/180:
        top_points[1,0] = base_points[1,0] - (top_points[1,1] * tan(pi*1/180))

    if abs(top_points[0,0] - base_points[0,0]) < JMC.geo_lentol or abs(top_points[1,0] - base_points[1,0]) < JMC.geo_lentol:
        # Side gradients are too large. This can still be the case if dimensions are small. Leave the co-ordinates as they are.
        None
    elif abs(top_points[0,1]) < JMC.geo_lentol or abs(top_points[1,1]) < JMC.geo_lentol:
        # One or both of the top points lie too close to the x axis. Leave the co-ordinates as they are.
        None
    else:
        # Make an approximation. Ensure that the side gradients are equal (although opposite).
        length_side1 = np.sqrt( (top_points[0,0]-base_points[0,0])**2 + (top_points[0,1]-base_points[0,1])**2 )
        length_side2 = np.sqrt( (top_points[1,0]-base_points[1,0])**2 + (top_points[1,1]-base_points[1,1])**2 )
        grad_side1 = (top_points[0,1] - base_points[0,1]) / (top_points[0,0] - base_points[0,0])
        grad_side2 = (top_points[1,1] - base_points[1,1]) / (top_points[1,0] - base_points[1,0])

        if abs(grad_side1) > 0 and abs(grad_side2) > 0:                                         # If either gradient is 0, leave the co-ordinates as they are
            if (length_side1 > length_side2 and length_side2/length_side1 > 0.5) or \
               (length_side2 > length_side1 and length_side1/length_side2 > 0.5):               # Sides have similar lengths. Set both to have the average gradient.
                grad = (abs(grad_side1) + abs(grad_side2)) / 2.0
                grad_side1 = grad; grad_side2 = - grad
            elif length_side1 < length_side2:                                                   # Side 1 much smaller than side 2. Set to have the side 2 gradient.
                grad_side1 = - grad_side2
            else:
                grad_side2 = - grad_side1                                                       # Side 2 much smaller than side 1. Set to have the side 1 gradient.
            top_points[0,0] = base_points[0,0] + (top_points[0,1] / grad_side1)                 # Adjust the top point x values
            top_points[1,0] = base_points[1,0] + (top_points[1,1] / grad_side2)

    return base_points, top_points

#############################################

def SaveEdgepointAndSectionFiles(source_name, edge_points1, edge_points2, section_parameters1, section_parameters2, section_perimeters1, section_perimeters2):

    """
    Saves the edge point and section files for each arm of the jet

    Parameters
    -----------
    source_name - str,
                  the name of the source

    edge_points1 - 2D array, shape(n,7)
                   Points on one arm of the jet, corresponding to the 
                   closest distance to that edge from the corresponding ridge point,
                   and their distance from source.

    edge_points2 - 2D array, shape(n,7)
                   Points on the other arm of the jet, corresponding to the 
                   closest distance to that edge from the corresponding ridge point,
                   and their distance from source.

    section_parameters1 - 2D array, shape(n,12)
                          Array for one arm of the jet, with section points (x/y * 4), 
                          distance from source and computed parameters

    section_parameters2 - 2D array, shape(n,12)
                          Array for other arm of the jet, with section points (x/y * 4), 
                          distance from source and computed parameters

    section_perimeters1 - 2D array, shape(n,pSize)
                          Array of perimeter co-ordinates for each merged section (-1 filled)
                          for one arm of the jet

    section_perimeters2 - 2D array, shape(n,pSize)
                          Array of perimeter co-ordinates for each merged section (-1 filled)
                          for other arm of the jet
    """

    fileEP1 = np.column_stack((edge_points1[:,0], edge_points1[:,1], edge_points1[:,2], edge_points1[:,3], edge_points1[:,4]))
    fileEP2 = np.column_stack((edge_points2[:,0], edge_points2[:,1], edge_points2[:,2], edge_points2[:,3], edge_points2[:,4]))
    np.savetxt(JSF.EP1 %(source_name, str(JMS.map_number+1)), fileEP1, delimiter=' ', \
               header='edgepoint x1-coord (pix), edgepoint y1-coord (pix), edgepoint x2-coord (pix), edgepoint y2-coord (pix), edgepoint R (pix)')
    np.savetxt(JSF.EP2 %(source_name, str(JMS.map_number+1)), fileEP2, delimiter=' ', \
               header='edgepoint x1-coord (pix), edgepoint y1-coord (pix), edgepoint x2-coord (pix), edgepoint y2-coord (pix), edgepoint R (pix)')

    fileSP1 = np.column_stack((section_parameters1[:,0], section_parameters1[:,1], section_parameters1[:,2], section_parameters1[:,3], \
                               section_parameters1[:,4], section_parameters1[:,5], section_parameters1[:,6], section_parameters1[:,7], \
                               section_parameters1[:,8], section_parameters1[:,9], section_parameters1[:,10], section_parameters1[:,11]))
    fileSP2 = np.column_stack((section_parameters2[:,0], section_parameters2[:,1], section_parameters2[:,2], section_parameters2[:,3], \
                               section_parameters2[:,4], section_parameters2[:,5], section_parameters2[:,6], section_parameters2[:,7], \
                               section_parameters2[:,8], section_parameters2[:,9], section_parameters2[:,10], section_parameters2[:,11]))
    np.savetxt(JSF.SP1 %(source_name, str(JMS.map_number+1)), fileSP1, delimiter=' ', \
               header='section x1-coord (pix), section y1-coord (pix), section x2-coord (pix), section y2-coord (pix), ' + \
                      'section x3-coord (pix), section y3-coord (pix), section x4-coord (pix), section y4-coord (pix), ' + \
                      'section R (pix), section flux (Jy/beam), section volume (pix**3), section area (pix**2)')
    np.savetxt(JSF.SP2 %(source_name, str(JMS.map_number+1)), fileSP2, delimiter=' ', \
               header='section x1-coord (pix), section y1-coord (pix), section x2-coord (pix), section y2-coord (pix), ' + \
                      'section x3-coord (pix), section y3-coord (pix), section x4-coord (pix), section y4-coord (pix), ' + \
                      'section R (pix), section flux (Jy/beam), section volume (pix**3), section area (pix**2)')

    maxlen = 0
    for sp in section_perimeters1:
        spCoordCnt = 0
        for sp_coord in sp:
            spCoordCnt += 1
            if isnan(sp_coord):
                if spCoordCnt > maxlen: maxlen = spCoordCnt     # maximum length of output record
                break
    fileSR1 = section_perimeters1[:,0:maxlen]
    maxlen = 0
    for sp in section_perimeters2:
        spCoordCnt = 0
        for sp_coord in sp:
            spCoordCnt += 1
            if isnan(sp_coord):
                if spCoordCnt > maxlen: maxlen = spCoordCnt     # maximum length of output record
                break
    fileSR2 = section_perimeters2[:,0:maxlen]
    np.savetxt(JSF.SR1 %(source_name, str(JMS.map_number+1)), fileSR1, delimiter=' ')
    np.savetxt(JSF.SR2 %(source_name, str(JMS.map_number+1)), fileSR2, delimiter=' ')

#############################################

def SaveRegions(source_name, section_perimeters1, section_perimeters2):

    """
    Save the section perimeter data as SAOImageDS9 regions.

    Parameters
    -----------
    source_name - str,
                  the name of the source

    section_perimeters1 - 2D array, shape(n,pSize)
                          Array of perimeter co-ordinates for each merged section (-1 filled)
                          for one arm of the jet

    section_perimeters2 - 2D array, shape(n,pSize)
                          Array of perimeter co-ordinates for each merged section (-1 filled)
                          for other arm of the jet
    """
    
    # Create a single perimeter array
    section_perimeters = np.vstack((section_perimeters1, section_perimeters2))

    # Initialise the list of all regions
    all_regions = []

    # Loop through all the section perimeters
    for sp in section_perimeters:

        # Loop through all sets of x/y co-ordinates
        spCoords = []
        spCoordCnt = 0
        for sp_coord in sp:
            spCoordCnt += 1

            if isnan(sp_coord):                         # End of perimeter co-ordinates
                break

            # Create 2D list of x/y co-ordinates
            if spCoordCnt % 2 != 0:
                spXY = [sp_coord]                       # x co-ordinate
            else:
                spXY.append(sp_coord)                   # y co-ordinate
                spCoords.append(spXY)

        # Convert perimeter x/y co-ordinate list to an array
        spCoordsArray = np.array(spCoords)

        # Convert to polygon and append polygon to list of regions
        spCoordsArray[:,0] += JMC.x_offset              # Translate co-ordinates (based on cutout) to full image co-ordinates
        spCoordsArray[:,1] += JMC.y_offset
        pix = PixCoord(x = spCoordsArray[:,0], y = spCoordsArray[:,1])
        all_regions.append(PolygonPixelRegion(pix))

    # Save as a DS9 region file
    Regions(all_regions).write(JSF.RGS %(source_name, str(JMS.map_number+1)), format='ds9')

#############################################

def PlotEdgePointsAndSections(flux_array, source_name, edge_points1, edge_points2, section_parameters1, section_parameters2):

    """
    Plots the edge points and jet sections on the source.
    Note, this requires the start and end section R values, to look for jumps.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    source_name - str,
                  the name of the source

    edge_points1 - 2D array, shape(n,7)
                   Points on one arm of the jet, corresponding to the 
                   closest distance to that edge from the corresponding ridge point,
                   and their distance from source.

    edge_points2 - 2D array, shape(n,7)
                   Points on the other arm of the jet, corresponding to the 
                   closest distance to that edge from the corresponding ridge point,
                   and their distance from source.

    section_parameters1 - 2D array, shape(n,12)
                          Array for one arm of the jet, with section points (x/y * 4), 
                          distance from source and computed parameters

    section_parameters2 - 2D array, shape(n,12)
                          Array for other arm of the jet, with section points (x/y * 4), 
                          distance from source and computed parameters
    """

    # Cut down map for plotting
    flux_array_plot = JMC.flux_factor * flux_array.copy()
    imCentre = np.array([ (edge_points1[0,0] + edge_points1[0,2]) / 2.0, (edge_points1[0,1] + edge_points1[0,3]) / 2.0 ])
    xmin = floor(imCentre[0] - (JMS.sSize[0] / 2.0)); xmax = floor(imCentre[0] + (JMS.sSize[0] / 2.0))
    ymin = floor(imCentre[1] - (JMS.sSize[1] / 2.0)); ymax = floor(imCentre[1] + (JMS.sSize[1] / 2.0))
    flux_array_plot  = flux_array_plot[ymin : ymax, xmin : xmax]

    palette = plt.cm.cividis
    palette = copy.copy(plt.get_cmap("cividis"))
    palette.set_bad('k',0.0)

    y, x = np.mgrid[slice((0),(flux_array_plot.shape[0]),1), slice((0),(flux_array_plot.shape[1]),1)]
    y = np.ma.masked_array(y, mask=np.ma.masked_invalid(flux_array_plot).mask)
    x = np.ma.masked_array(x, mask=np.ma.masked_invalid(flux_array_plot).mask)

    x_pixels_not_plotted = (1.0 - JMS.ImFraction) * float(JMS.sSize[0])
    xplotmin = x_pixels_not_plotted / 2.0
    xplotmax = float(JMS.sSize[0]) - (x_pixels_not_plotted / 2.0)
    y_pixels_not_plotted = (1.0 - JMS.ImFraction) * float(JMS.sSize[1])
    yplotmin = y_pixels_not_plotted / 2.0
    yplotmax = float(JMS.sSize[1]) - (y_pixels_not_plotted / 2.0)

    # Plot edge points
    fig, ax = plt.subplots(figsize=(10,10))
    fig.suptitle('Source: %s' %source_name)
    fig.subplots_adjust(top=0.9)
    ax.set_aspect('equal')
    ax.set_xlim(xplotmin, xplotmax)
    ax.set_ylim(yplotmin, yplotmax)
    
    A = np.ma.array(flux_array_plot, mask=np.ma.masked_invalid(flux_array_plot).mask)
    c = ax.pcolor(x, y, A, cmap=palette, vmin=JMS.vmin, vmax=JMS.vmax)

    # Setup offsets for cutdown map
    edge_plot1 = edge_points1.copy(); edge_plot1[:,0] -= xmin;  edge_plot1[:,2] -= xmin;  edge_plot1[:,1] -= ymin; edge_plot1[:,3] -= ymin
    edge_plot2 = edge_points2.copy(); edge_plot2[:,0] -= xmin;  edge_plot2[:,2] -= xmin;  edge_plot2[:,1] -= ymin; edge_plot2[:,3] -= ymin
    section_plot1 = section_parameters1.copy(); section_plot1[:,0] -= xmin; section_plot1[:,2] -= xmin; section_plot1[:,4] -= xmin
    section_plot1[:,6] = np.where(section_plot1[:,6] != -1, section_plot1[:,6] - xmin, section_plot1[:,6])
    section_plot1[:,1] -= ymin; section_plot1[:,3] -= ymin; section_plot1[:,5] -= ymin
    section_plot1[:,7] = np.where(section_plot1[:,7] != -1, section_plot1[:,7] - ymin, section_plot1[:,7])
    section_plot2 = section_parameters2.copy(); section_plot2[:,0] -= xmin; section_plot2[:,2] -= xmin; section_plot2[:,4] -= xmin
    section_plot2[:,6] = np.where(section_plot2[:,6] != -1, section_plot2[:,6] - xmin, section_plot2[:,6])
    section_plot2[:,1] -= ymin; section_plot2[:,3] -= ymin; section_plot2[:,5] -= ymin
    section_plot2[:,7] = np.where(section_plot2[:,7] != -1, section_plot2[:,7] - ymin, section_plot2[:,7])

    epcount = 0
    for ep in edge_plot1:
        epcount += 1
        x_values = np.array([ep[0], ep[2]])
        y_values = np.array([ep[1], ep[3]])
        ax.plot(x_values, y_values, 'y-', linewidth=0.6)              # Edge point segment separators
        if epcount > 1:
            if (ep[4] - last_ep[4]) < (JMC.MaxRFactor * JMC.R_es):    # If gap is too big, don't draw edge lines 
                x_values = np.array([ep[0], last_ep[0]])
                y_values = np.array([ep[1], last_ep[1]])
                ax.plot(x_values, y_values, 'r-', linewidth=0.6)      # Edge line 1
                x_values = np.array([ep[2], last_ep[2]])
                y_values = np.array([ep[3], last_ep[3]])
                ax.plot(x_values, y_values, 'r-', linewidth=0.6)      # Edge line 2
            else:
                ep_count = 1    # reset counter
        last_ep = ep

    epcount = 0
    for ep in edge_plot2:
        epcount += 1
        x_values = np.array([ep[0], ep[2]])
        y_values = np.array([ep[1], ep[3]])
        ax.plot(x_values, y_values, 'y-', linewidth=0.6)              # Edge point segment separators
        if epcount > 1:
            if (ep[4] - last_ep[4]) < (JMC.MaxRFactor * JMC.R_es):    # If gap is too big, don't draw edge lines 
                x_values = np.array([ep[0], last_ep[0]])
                y_values = np.array([ep[1], last_ep[1]])
                ax.plot(x_values, y_values, 'r-', linewidth=0.6)      # Edge line 1
                x_values = np.array([ep[2], last_ep[2]])
                y_values = np.array([ep[3], last_ep[3]])
                ax.plot(x_values, y_values, 'r-', linewidth=0.6)      # Edge line 2
            else:
                ep_count = 1    # reset counter
        last_ep = ep

    # Plot centre line in red, to make it obvious
    ep = edge_plot1[0,:]
    x_values = np.array([ep[0], ep[2]])
    y_values = np.array([ep[1], ep[3]])
    ax.plot(x_values, y_values, 'r-', linewidth=0.8)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(c, cax = cax)
    
    fig.savefig(JSF.EPimage %(source_name, str(JMS.map_number+1)))
    plt.close(fig)

    # Plot jet sections
    fig, ax = plt.subplots(figsize=(10,10))
    fig.suptitle('Source: %s' %source_name)
    fig.subplots_adjust(top=0.9)
    ax.set_aspect('equal')
    ax.set_xlim(xplotmin, xplotmax)
    ax.set_ylim(yplotmin, yplotmax)
    
    A = np.ma.array(flux_array_plot, mask=np.ma.masked_invalid(flux_array_plot).mask)
    c = ax.pcolor(x, y, A, cmap=palette, vmin=JMS.vmin, vmax=JMS.vmax)

    spcount = 0
    for sp in section_plot1:
        spcount += 1
        x_values = np.array([sp[0], sp[2]])
        y_values = np.array([sp[1], sp[3]])
        ax.plot(x_values, y_values, 'y-', linewidth=0.6)                                # Segment separators
        if spcount > 1 and (sp[8] - last_sp[9]) > (JMC.MaxRFactor * JMC.R_es):          # If gap is too big, plot last separator
            PlotLastSegment(ax, last_sp)
        last_sp = sp
    PlotLastSegment(ax, sp)                                                             # Plot last separator

    spcount = 0
    for sp in section_plot2:
        spcount += 1
        x_values = np.array([sp[0], sp[2]])
        y_values = np.array([sp[1], sp[3]])
        ax.plot(x_values, y_values, 'y-', linewidth=0.6)                                # Segment separators
        if spcount > 1 and (sp[8] - last_sp[9]) > (JMC.MaxRFactor * JMC.R_es):          # If gap is too big, plot last separator
            PlotLastSegment(ax, last_sp)
        last_sp = sp
    PlotLastSegment(ax, sp)                                                             # Plot last separator

    # Plot centre line in red, to make it obvious
    sp = section_plot1[0,:]
    x_values = np.array([sp[0], sp[2]])
    y_values = np.array([sp[1], sp[3]])
    ax.plot(x_values, y_values, 'r-', linewidth=0.8)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(c, cax = cax)
    
    fig.savefig(JSF.SCimage %(source_name, str(JMS.map_number+1)))
    plt.close(fig)

#############################################

def PlotLastSegment(ax, sp):

    """
    Plots the edge points and jet sections on the source.

    Parameters
    -----------
    sp - 1D array
         section points (x/y * 4), distance from source 
         and computed parameters for last segment

    ax - plot axes
    """

    # Check if this is a 3-point segment
    if sp[6] == -1:
        ax.plot(np.array([sp[4]-0.1, sp[4]+0.1]), np.array([sp[5]-0.1, sp[5]+0.1]), 'g-', linewidth=0.6)     # Plot a small line around the 3rd point
    else:
        ax.plot(np.array([sp[4], sp[6]]), np.array([sp[5], sp[7]]), 'g-', linewidth=0.6)

#############################################

def PolarCoordinates(a, pos):

    """
    Returns a grid of r and phi polar coordinate values
    corresponding to each pixel, with the origin position
    provided as a parameter

    Parameters
    -----------

    a - 2D array, shape(n,2)
        array of pixels to be mapped onto polar coordinates

    pos - 1D array
          central point position given as pixel
          values [x_position, y_position]

    Returns
    -----------

    r - 2D array, shape(a.shape)
        array of r coordinate values corresponding to each pixel
        in provided array a

    phi - 2D array, shape(a.shape)
          array of phi coordinate values corresponding to each
          pixel in provided array a
    """

    X = pos[0]
    Y = pos[1]
    r = np.zeros_like(a)
    phi = np.zeros_like(a)
    a_offset = np.indices(a.shape)

    r[:,:] = np.sqrt((a_offset[0,:,:] + 0.5 - Y)**2 + (a_offset[1,:,:] + 0.5 - X)**2)
    phi[:,:] = np.round(np.arctan2((a_offset[0,:,:] + 0.5 - Y), (a_offset[1,:,:] + 0.5 - X)), 3)

    return r, phi

#############################################

def CheckQuadrant(angle):

    """
    Returns angle quadrant for given input angle

    Parameters
    -----------

    angle - float,
            angle of interest, given in radians

    Returns
    -----------

    quadrant - int,
               quadrant, in which the angle is located.
    """

    angle_round = np.round(angle,3)
    pi_round = np.round(pi,3)

    if 0 <= angle_round <= pi_round/2.:
        quadrant = 1

    elif pi_round/2. <= angle_round <= pi_round:
        quadrant = 2

    elif -pi_round <= angle_round <= -pi_round/2.:
        quadrant = 3

    elif -pi_round/2. <= angle_round <= 0:
        quadrant = 4

    return quadrant

#############################################

def PiRange(angle):

    """
    Ensures the angle belongs to [-pi,pi] range

    Parameters
    -----------

    angle - float,
            angle of interest, given in radians

    Returns
    -----------

    angle_out - float,
                input angle in the range [-pi,pi]
    """

    if angle < -pi:
        angle_out = angle + 2*pi

    elif angle > pi:
        angle_out = angle - 2*pi

    else:
        angle_out = angle

    return angle_out

#############################################
        
def TwoPiRange(angle):

    """
    Ensures the angle belongs to [0,2pi] range

    Parameters
    -----------

    angle - float,
            angle of interest, given in radians

    Returns
    -----------

    angle_out - float,
                input angle in the range [0,2pi]
    """

    if angle < 0:
        angle_out = 2*pi + angle

    else:
        angle_out = angle

    return angle_out

#############################################