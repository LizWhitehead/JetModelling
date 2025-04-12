#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EdgeToolkit.py
Toolkit for edge-point finding in a source
Created by LizWhitehead - Jan 2025
"""

import JetRidgeline.RidgelineFiles as RLF
import JetRidgeline.RLConstants as RLC
import JetRidgeline.RLGlobal as RLG
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask
import numpy as np
from math import tan, atan2, pow
from numpy import pi, sin, cos, dot
import copy

#############################################

def FindEdgePoints(area_fluxes, ridge_point, ridge_phi, ridge_R, prev_edge_points):

    """
    Returns edge points, on either side of the jet, corresponding to
    the closest distance to that edge from the supplied ridge point.

    Parameters
    -----------
    area_fluxes - 2D array,
                  raw image array

    ridge_point - 1D array, shape(2,)
                  ridge point along the ridge line

    ridge_phi - float,
                angular direction of the ridge line

    ridge_R - float,
              distance from the source along the jet in pixels

    prev_edge_points - 1D array, shape(5,)
                       edge points corresponding to
                       previous ridge point and distance from source
                       (x1,y1,x2,y2,R)
    
    Constants
    ---------

    Returns
    -----------
    stop_finding_edge_points - flag for no further edge points to be located

    edge_points - 1D array, shape(5,)
                  Points on either side of the jet, corresponding to the 
                  closest distance to that edge from the corresonding ridge point,
                  and distance from source
                  (x1,y1,x2,y2,R)

    Notes
    -----------
    """

    # Initialise flag to indicate that edge points determination should stop
    stop_finding_edge_points = False

    # Initialise edge_points array
    edge_points = np.full((1,5), np.nan)

    # Get polar coordinates for area_fluxes, around the ridge point
    r, phi = PolarCoordinates(area_fluxes, ridge_point)

    # Find the r and phi values of the previous edge points
    prev_edge_pix = np.floor(prev_edge_points).astype('int')
    phi_prev_coord1 = phi[prev_edge_pix[1], prev_edge_pix[0]]; phi_prev_coord2 = phi[prev_edge_pix[3], prev_edge_pix[2]]
    r_prev_coord1 = r[prev_edge_pix[1], prev_edge_pix[0]]; r_prev_coord2 = r[prev_edge_pix[3], prev_edge_pix[2]]

    # Get the angle quadrants
    quad_ridge_phi = CheckQuadrant(ridge_phi)
    quad_phi_coord1 = CheckQuadrant(phi_prev_coord1)
    quad_phi_coord2 = CheckQuadrant(phi_prev_coord2)

    # Find the difference between phi of the ridge point and that of each previous edge point
    phi_diff1 = np.abs(phi_prev_coord1 - ridge_phi)
    min_phi = min(phi_prev_coord1, ridge_phi); max_phi = max(phi_prev_coord1, ridge_phi)
    quad_min = CheckQuadrant(min_phi); quad_max = CheckQuadrant(max_phi)
    diff = max_phi - min_phi
    if 3 <= quad_min <= 4 and 1 <= quad_max <= 2 and diff > pi: phi_diff1 = np.abs(phi_diff1 - 2*pi)
    phi_diff2 = np.abs(phi_prev_coord2 - ridge_phi)
    min_phi = min(phi_prev_coord2, ridge_phi); max_phi = max(phi_prev_coord2, ridge_phi)
    quad_min = CheckQuadrant(min_phi); quad_max = CheckQuadrant(max_phi)
    diff = max_phi - min_phi
    if 3 <= quad_min <= 4 and 1 <= quad_max <= 2 and diff > pi: phi_diff2 = np.abs(phi_diff2 - 2*pi)

    # Start looking for the edge point at the side closest to the ridge point
    if r_prev_coord1 < r_prev_coord2:
        r_prev1 = r_prev_coord1; r_prev2 = r_prev_coord2
        phi_prev1 = phi_prev_coord1; phi_prev2 = phi_prev_coord2
        quad_phi_prev1 = quad_phi_coord1; quad_phi_prev2 = quad_phi_coord2
        phi_diff_ridge_prev1 = phi_diff1; phi_diff_ridge_prev2 = phi_diff2
    else:
        r_prev1 = r_prev_coord2; r_prev2 = r_prev_coord1
        phi_prev1 = phi_prev_coord2; phi_prev2 = phi_prev_coord1
        quad_phi_prev1 = quad_phi_coord2; quad_phi_prev2 = quad_phi_coord1
        phi_diff_ridge_prev1 = phi_diff2; phi_diff_ridge_prev2 = phi_diff1

    # Fill invalid (nan) values with zeroes in area_fluxes
    area_fluxes_valid = np.ma.filled(np.ma.masked_invalid(area_fluxes), 0)

    # Mask the jet - where flux values are above (RLC.nSig * rms)
    jet_mask = np.ma.masked_where(area_fluxes_valid > (RLC.nSig * RLG.bgRMS), area_fluxes_valid).mask

    # Create a mask to search for nearest edge point on one side of the ridge point
    min_phi = min(ridge_phi, phi_prev1); max_phi = max(ridge_phi, phi_prev1)
    quad_min = CheckQuadrant(min_phi); quad_max = CheckQuadrant(max_phi)
    diff = max_phi - min_phi
    if 3 <= quad_min <= 4 and 1 <= quad_max <= 2 and diff > pi:
        phi_mask = np.ma.masked_inside(phi, ridge_phi, phi_prev1).mask              # search between ridge_phi and phi_prev1
    else:
        phi_mask = np.ma.masked_outside(phi, ridge_phi, phi_prev1).mask
    prev_edge_mask = np.ma.masked_where(phi == phi_prev1, phi).mask                 # mask the previous edge point phi
    search_mask = np.ma.mask_or(np.ma.mask_or(phi_mask, prev_edge_mask), jet_mask)  # search outside the jet

    # Find the co-ordinate of the smallest r value in the search area - the first edge point
    r_search = np.ma.masked_array(r, mask = search_mask, copy = True)
    edge_coord1_yx = np.unravel_index(np.argmin(r_search, axis=None), r_search.shape)
    edge_coord1 = np.array([edge_coord1_yx[1] + 0.5,edge_coord1_yx[0] + 0.5])

    # If the ridge point is very close to an edge of the jet, the second edge point
    # could be found on the same side of the jet. Reduce the angle of search until the
    # second edge point is found on the other side of the jet.

    edge_coord2 = np.full((1,2), np.nan)    # Initialise 2nd edge point array
    start_phi = ridge_phi                   # Initialise the start search angle as ridge_phi
    quad_start_phi = quad_ridge_phi
    phi_diff_start_prev2 = phi_diff_ridge_prev2

    # Loop to find the second edge point. 
    # Reduce the search angle by 10 degrees if not found on the other side of the jet.
    while np.isnan(edge_coord2).any() and phi_diff_start_prev2 >= (pi*10/180):

        # Create a mask to search for nearest edge point on the other side of the ridge point
        min_phi = min(start_phi, phi_prev2); max_phi = max(start_phi, phi_prev2)
        quad_min = CheckQuadrant(min_phi); quad_max = CheckQuadrant(max_phi)
        diff = max_phi - min_phi
        if 3 <= quad_min <= 4 and 1 <= quad_max <= 2 and diff > pi:
            phi_mask = np.ma.masked_inside(phi, start_phi, phi_prev2).mask              # search between start_phi and phi_prev2
        else:
            phi_mask = np.ma.masked_outside(phi, start_phi, phi_prev2).mask
        prev_edge_mask = np.ma.masked_where(phi == phi_prev2, phi).mask                 # mask the previous edge point phi
        search_mask = np.ma.mask_or(np.ma.mask_or(phi_mask, prev_edge_mask), jet_mask)  # search outside the jet

        # Find the co-ordinate of the smallest r value in the search area - the second edge point
        r_search = np.ma.masked_array(r, mask = search_mask, copy = True)
        edge_coord2_yx = np.unravel_index(np.argmin(r_search, axis=None), r_search.shape)
        phi_coord2 = phi[edge_coord2_yx]

        # Test if this is on the same side of the jet as the first edge point. 

        # Compare the distance between this edge point and the first edge point, with the distance 
        # between this edge point and the last corresponding edge point.
        r2, phi2 = PolarCoordinates(area_fluxes, np.array([edge_coord2_yx[1],edge_coord2_yx[0]]))       # polar coordinates around the second edge point
        r_coord1 = r2[edge_coord1_yx]
        if phi_prev1 == phi_prev_coord1:
            r_last_edge2 = r2[prev_edge_pix[3], prev_edge_pix[2]]
        else:
            r_last_edge2 = r2[prev_edge_pix[1], prev_edge_pix[0]]
        
        # Take polar coordinates around the previous edge point, corresponding to the first found edge point.
        # From here, look at the difference in angles of the first and and second edge points.
        # This will be small if the edge points are on the same side of the jet.
        if phi_prev1 == phi_prev_coord1:
            r3, phi3 = PolarCoordinates(area_fluxes, np.array([prev_edge_pix[0], prev_edge_pix[1]]))    # polar coordinates around the last edge point 1
        else:
            r3, phi3 = PolarCoordinates(area_fluxes, np.array([prev_edge_pix[2], prev_edge_pix[3]]))
        phi_last_edge1_coord1 = phi3[edge_coord1_yx]
        phi_last_edge1_coord2 = phi3[edge_coord2_yx]
        phi_diff = np.abs(phi_last_edge1_coord1 - phi_last_edge1_coord2)
        min_phi = min(phi_prev_coord1, ridge_phi); max_phi = max(phi_prev_coord1, ridge_phi)
        quad_min = CheckQuadrant(min_phi); quad_max = CheckQuadrant(max_phi)
        diff = max_phi - min_phi
        if 3 <= quad_min <= 4 and 1 <= quad_max <= 2 and diff > pi: phi_diff = np.abs(phi_diff - 2*pi)

        if r_coord1 < r_last_edge2 or phi_diff < (pi*20/180):       # Test whether we think the edge points are on the same side of the jet     
            # Reduce the angle of search by 10 degrees
            phi_start_minus10 = PiRange(start_phi - (pi*10/180))          # Try taking away
            phi_diff_start_minus10 = np.abs(phi_start_minus10 - phi_prev2)
            min_phi = min(phi_start_minus10, phi_prev2); max_phi = max(phi_start_minus10, phi_prev2)
            quad_min = CheckQuadrant(min_phi); quad_max = CheckQuadrant(max_phi)
            diff = max_phi - min_phi
            if 3 <= quad_min <= 4 and 1 <= quad_max <= 2 and diff > pi: phi_diff_start_minus10 = np.abs(phi_diff_start_minus10 - 2*pi)

            if phi_diff_start_minus10 < phi_diff_start_prev2:
                start_phi = phi_start_minus10                       # Angle successfully reduced
                phi_diff_start_prev2 = phi_diff_start_minus10
            else:
                start_phi = PiRange(start_phi + (pi*10/180))              # Angle not reduced, so add
                phi_diff_start_plus10 = np.abs(start_phi - phi_prev2)
                min_phi = min(start_phi, phi_prev2); max_phi = max(start_phi, phi_prev2)
                quad_min = CheckQuadrant(min_phi); quad_max = CheckQuadrant(max_phi)
                diff = max_phi - min_phi
                if 3 <= quad_min <= 4 and 1 <= quad_max <= 2 and diff > pi: phi_diff_start_plus10 = np.abs(phi_diff_start_plus10 - 2*pi)
                phi_diff_start_prev2 = phi_diff_start_plus10

            quad_start_phi = CheckQuadrant(start_phi)
        else:
            # Second edge point has been found on the other side of the jet
            edge_coord2 = np.array([edge_coord2_yx[1] + 0.5,edge_coord2_yx[0] + 0.5])

    if not np.isnan(edge_coord2).any():
        # Detemine which should be the first co-ordinate in the array.
        if phi_prev1 == phi_prev_coord1:
            edge_points = np.array([edge_coord1[0], edge_coord1[1], edge_coord2[0], edge_coord2[1], ridge_R])
        else:
            edge_points = np.array([edge_coord2[0], edge_coord2[1], edge_coord1[0], edge_coord1[1], ridge_R])

    return stop_finding_edge_points, edge_points

#############################################

def FindInitEdgePoints(area_fluxes, ridge_point, ridge_phi, ridge_R):

    """
    Returns edge points, on either side of the jet, corresponding to
    the closest distance to that edge from the initial ridge point.

    Parameters
    -----------
    area_fluxes - 2D array,
                  raw image array

    ridge_point - 1D array, shape(2,)
                  ridge point along the ridge line

    ridge_phi - float,
                angular direction of the ridge line

    ridge_R - float,
              distance from the source along the jet in pixels
    
    Constants
    ---------

    Returns
    -----------
    edge_points - 2D array, shape(5)
                  Points on either side of the jet, corresponding to the 
                  closest distance to that edge from the initial ridge point,
                  and their distance from source.
                  (x1,y1,x2,y2,R)

    Notes
    -----------
    """

    # Initialise edge_points array
    edge_points = np.full((1,5), np.nan)

    # Get polar coordinates for area_fluxes, around the ridge point
    r, phi = PolarCoordinates(area_fluxes, ridge_point)

    # Fill invalid (nan) values with zeroes in area_fluxes
    area_fluxes_valid = np.ma.filled(np.ma.masked_invalid(area_fluxes), 0)

    # Mask the jet - where flux values are above (RLC.nSig * rms)
    jet_mask = np.ma.masked_where(area_fluxes_valid > (RLC.nSig * RLG.bgRMS), area_fluxes_valid).mask

    # Search for an edge at right angles to the ridge direction
    search_phi_range1 = PiRange(ridge_phi + (pi*90/180) - (pi*5/180))     # +/- 5 degrees
    search_phi_range2 = PiRange(ridge_phi + (pi*90/180) + (pi*5/180))
    if CheckQuadrant(search_phi_range2) == 3 and CheckQuadrant(search_phi_range1) == 2:
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
    if CheckQuadrant(search_phi_range2) == 3 and CheckQuadrant(search_phi_range1) == 2:
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

    # Add to the edge points array in order of phi value
    if phi[edge_coord1_yx] < phi[edge_coord2_yx]:
        edge_points = np.array([edge_coord1[0], edge_coord1[1], edge_coord2[0], edge_coord2[1], ridge_R])
    else:
        edge_points = np.array([edge_coord2[0], edge_coord2[1], edge_coord1[0], edge_coord1[1], ridge_R])

    return edge_points

#############################################

def AddEdgePoints(area_fluxes, edge_points):

    """
    Interpolates extra edge points at points in the jet where
    a long edge line cuts off parts of the jet flux.

    Parameters
    -----------
    area_fluxes - 2D array, shape(n,2)
                  raw image array

    edge_points - 2D array, shape(n,4)
                  Points on one arm of the jet, corresponding to the 
                  closest distance to that edge from the corresponding ridge point.
    
    Constants
    ---------

    Returns
    -----------
    edge_points_updated - 2D array, shape(n,5)
                          Array with extra interpolated edge points
                          and their distance from source

    Notes
    -----------
    """

    # Initialise edgepoints array
    edge_points_updated = edge_points

    # Fill invalid (nan) values with zeroes in area_fluxes
    area_fluxes_valid = np.ma.filled(np.ma.masked_invalid(area_fluxes), 0)

    # Mask the jet - where flux values are above (RLC.nSig * rms)
    jet_mask = np.ma.masked_where(area_fluxes_valid > (RLC.nSig * RLG.bgRMS), area_fluxes_valid).mask

    # Loop for each side of the jet
    jet_side = 1
    while jet_side <= 2:

        # Loop down this side and look for long edgelines, which will be likely to cut-off flux
        point_count = 0
        for [x_side1, y_side1, x_side2, y_side2, ridge_R] in edge_points_updated:
            point_count += 1

            x_side1_floor = np.floor(x_side1); y_side1_floor = np.floor(y_side1)
            x_side2_floor = np.floor(x_side2); y_side2_floor = np.floor(y_side2)

            if point_count > 1:
                if jet_side == 1:
                    dist = np.sqrt( (x_side1_floor - lastpts[0])**2 + (y_side1_floor - lastpts[1])**2 ) # distance between last 2 points on this side
                else:
                    dist = np.sqrt( (x_side2_floor - lastpts[2])**2 + (y_side2_floor - lastpts[3])**2 ) # distance between last 2 points on this side

                R_diff = ridge_R - lastpts[4]   # change in R since last edgepoint

                if dist > (RLC.MinIntpolFactor * RLC.R):
                    # Distance is long enough to have cut off some flux.
                    # Work out the number of sections to divide the length between last 2 points.
                    num_sections = max( min(np.floor(dist/RLC.R + 1).astype('int'), RLC.MaxIntpolSections), 2)

                    R_per_section = R_diff / num_sections                               # change in R per section

                    # Interpolate the new edgepoint for each section
                    section_x1_length = (x_side1_floor - lastpts[0]) / num_sections     # x section length on one side of the jet
                    section_y1_length = (y_side1_floor - lastpts[1]) / num_sections     # y section length on one side of the jet
                    section_x2_length = (x_side2_floor - lastpts[2]) / num_sections     # x section length on other side of the jet
                    section_y2_length = (y_side2_floor - lastpts[3]) / num_sections     # y section length on other side of the jet

                    last_sect_x1 = lastpts[0]; last_sect_y1 = lastpts[1]
                    last_sect_x2 = lastpts[2]; last_sect_y2 = lastpts[3]
                    sect = 1
                    while sect <= (num_sections-1):
                        R_section = lastpts[4] + (R_per_section * sect)                 # R for this section

                        x1_float = last_sect_x1 + section_x1_length; y1_float = last_sect_y1 + section_y1_length
                        x2_float = last_sect_x2 + section_x2_length; y2_float = last_sect_y2 + section_y2_length

                        if section_x1_length >= 0: x1 = np.floor(x1_float).astype('int')
                        else: x1 = np.ceil(x1_float).astype('int')                          # x section co-ord on one side of the jet
                        if section_y1_length >= 0: y1 = np.floor(y1_float).astype('int')
                        else: y1 = np.ceil(y1_float).astype('int')                          # y section co-ord on one side of the jet
                        if section_x2_length >= 0: x2 = np.floor(x2_float).astype('int')
                        else: x2 = np.ceil(x2_float).astype('int')                          # x section co-ord on other side of the jet
                        if section_y2_length >= 0: y2 = np.floor(y2_float).astype('int')
                        else: y2 = np.ceil(y2_float).astype('int')                          # y section co-ord on other side of the jet

                        x_mean_float = (x1_float + x2_float) / 2; y_mean_float = (y1_float + y2_float) / 2
                        x_mean = np.floor(x_mean_float).astype('int')   # x co-ord of centre of the points on either side of the jet
                        y_mean = np.floor(y_mean_float).astype('int')   # y co-ord of centre of the points on either side of the jet

                        r, phi = PolarCoordinates(area_fluxes, np.array([x_mean,y_mean])) # polar co-ordinates around centre point

                        # Set up the start and end of the phi range around the section point
                        if jet_side == 1:
                            x_start = np.floor(x1_float - section_x1_length/2).astype('int'); y_start = np.floor(y1_float - section_y1_length/2).astype('int')
                            x_end = np.floor(x1_float + section_x1_length/2).astype('int'); y_end = np.floor(y1_float + section_y1_length/2).astype('int')
                            phi_start = phi[y_start,x_start]
                            phi_end = phi[y_end,x_end]
                        else:
                            x_start = np.floor(x2_float - section_x2_length/2).astype('int'); y_start = np.floor(y2_float - section_y2_length/2).astype('int')
                            x_end = np.floor(x2_float + section_x2_length/2).astype('int'); y_end = np.floor(y2_float + section_y2_length/2).astype('int')
                            phi_start = phi[y_start,x_start]
                            phi_end = phi[y_end,x_end]

                        min_phi = min(phi_start, phi_end); max_phi = max(phi_start, phi_end)
                        quad_min = CheckQuadrant(min_phi); quad_max = CheckQuadrant(max_phi)
                        diff = max_phi - min_phi
                        if 3 <= quad_min <= 4 and 1 <= quad_max <= 2 and diff > pi:
                            phi_mask = np.ma.masked_inside(phi, phi_start, phi_end).mask    # search between phi_start and phi_end
                        else:
                            phi_mask = np.ma.masked_outside(phi, phi_start, phi_end).mask
                        search_mask = np.ma.mask_or(phi_mask, jet_mask)                     # search outside the jet

                        # Find the co-ordinate of the smallest r value in the search area - the interpolated point
                        r_search = np.ma.masked_array(r, mask = search_mask, copy = True)
                        edgepoint_yx = np.unravel_index(np.argmin(r_search, axis=None), r_search.shape)

                        # Add interpolated edge points to the array
                        if jet_side == 1:
                            added_edgepoint_coords = np.array([edgepoint_yx[1] + 0.5,edgepoint_yx[0] + 0.5, x2 + 0.5, y2 + 0.5, R_section])
                        else:
                            added_edgepoint_coords = np.array([x1 + 0.5, y1 + 0.5, edgepoint_yx[1] + 0.5, edgepoint_yx[0] + 0.5, R_section])
                        edge_points_side = np.vstack((edge_points_side, added_edgepoint_coords))

                        last_sect_x1 = last_sect_x1 + section_x1_length; last_sect_y1 = last_sect_y1 + section_y1_length
                        last_sect_x2 = last_sect_x2 + section_x2_length; last_sect_y2 = last_sect_y2 + section_y2_length
                        sect += 1

                edge_points_side = np.vstack((edge_points_side, np.array([x_side1, y_side1, x_side2, y_side2, ridge_R])))
            else:
                edge_points_side = np.array([x_side1, y_side1, x_side2, y_side2, ridge_R])

            lastpts = np.array([x_side1_floor, y_side1_floor, x_side2_floor, y_side2_floor, ridge_R])

        edge_points_updated = edge_points_side
        jet_side += 1

    return edge_points_updated

#############################################

def GetJetSections(area_fluxes, edge_points1, edge_points2):

    """
    Get parameters for each section of the jet.

    Parameters
    -----------
    area_fluxes - 2D array,
                  raw image array

    edge_points1 - 2D array, shape(n,5)
                   Points on one arm of the jet, corresponding to the 
                   closest distance to that edge from the corresponding ridge point,
                   and their distance from source.

    edge_points2 - 2D array, shape(n,5)
                   Points on other arm of the jet, corresponding to the 
                   closest distance to that edge from the corresponding ridge point,
                   and their distance from source.
    
    Constants
    ---------

    Returns
    -----------
    section_params_merged1 - 2D array, shape(n,12)
                             Array with section points (x,y * 4), distance from source
                             and computed parameters for one arm of the jet

    section_params_merged2 - 2D array, shape(n,12)
                             Array with section points (x,y * 4), distance from source
                             and computed parameters for other arm of the jet

    Notes
    -----------
    """

    # Fill invalid (nan) values with zeroes in area_fluxes
    area_fluxes_valid = np.ma.filled(np.ma.masked_invalid(area_fluxes), 0)

    # Get the section polygon points
    polygon_points1 = GetSectionPolygons(edge_points1)
    polygon_points2 = GetSectionPolygons(edge_points2)

    # Get flux and volume for each arm of the jet
    section_parameters1 = GetSectionParameters(area_fluxes_valid, polygon_points1, initial_polygon_points = polygon_points2[0,0:8])
    section_parameters2 = GetSectionParameters(area_fluxes_valid, polygon_points2, initial_polygon_points = polygon_points1[0,0:8])

    # Merge sections to within a required count range for each arm of the jet
    section_params_merged1 = MergeSections(section_parameters1)
    section_params_merged2 = MergeSections(section_parameters2)

    return section_params_merged1, section_params_merged2

#############################################

def MergeSections(section_parameters):

    """
    Merge sections to a required number for one arm of the jet

    Parameters
    -----------
    section_parameters - 2D array, shape(n,12)
                         Array with section points (x,y * 4), distance from source
                         and computed parameters for one arm of the jet
    
    Constants
    ---------

    Returns
    -----------
    section_params_merged - 2D array, shape(n,12)
                            Array with merged section points (x,y * 4), distance from
                            source and computed parameters for one arm of the jet

    Notes
    -----------
    """

    # Initialise merged section parameters array
    section_params_merged = np.empty((0,12))

    # Initialise the first section flux
    first_section_flux = section_parameters[0,10]

    # Iterate, up to a maximum number of times to try to get the number of merged sections to the required value
    iteration_count = 0
    while ((np.size(section_params_merged, 0) < RLC.MinSectionsPerArm) or \
           (np.size(section_params_merged, 0) > RLC.MaxSectionsPerArm)) and iteration_count < RLC.MaxMergeIterations:
        iteration_count += 1

        # Set the max flux per merged section for this iteration
        if np.size(section_params_merged, 0) < RLC.MinSectionsPerArm:
            # Reduce maximum flux by % for each iteration
            max_flux = first_section_flux - (first_section_flux * (iteration_count-1) * RLC.PercChangeInMaxFlux/100)
        else:
            # Increase maximum flux by % for each iteration
            max_flux = first_section_flux + (first_section_flux * (iteration_count-1) * RLC.PercChangeInMaxFlux/100)

        section_params_merged = np.empty((0,12))                            # Re-initialise merged section parameters array
        last_merged_sections = np.full((1,12), np.nan)                      # Initialise last merged sections array
        last_merged_flux_section = 0.0; last_merged_volume_section = 0.0    # Initialise last merged flux/volume values

        # Loop around all sections and merge while total flux is less than the 
        # maximum value (the flux of the first section)
        sect_count = 0
        for [x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, flux_section, volume_section] in section_parameters:
            sect_count += 1

            merged_flux = last_merged_flux_section + flux_section               # Add on the flux in the latest section
            merged_volume = last_merged_volume_section + volume_section         # Add on the volume in the latest section

            # Setup the merged section vertices co-ordinates, for testing against the beam size
            if np.isnan(last_merged_sections).any():
                merged_section_coords = np.array([x1,y1, x2,y2, x3,y3, x4,y4])
            else:
                merged_section_coords = np.array([last_merged_sections[0],last_merged_sections[1], 
                                                  last_merged_sections[2],last_merged_sections[3], x3,y3, x4,y4])

            # Test whether maximum flux achieved.
            # Take the first section separately, as this is used for the initial maximum flux value.
            # But the merged section must ALWAYS be larger than the beam size.
            if (merged_flux >= max_flux or sect_count == 1) and LargerThanBeamSize(merged_section_coords):
                # Adding in this section flux would exceed the maximum flux
                if np.isnan(last_merged_sections).any():
                    # This section on its own exceeds the maximum flux. Add to the merged section parameters array.
                    section_params_merged = np.vstack((section_params_merged, \
                                        np.array([x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, flux_section, volume_section])))
                    last_merged_sections = np.full((1,12), np.nan)                              # Reset last merged sections array
                    last_merged_flux_section = 0.0; last_merged_volume_section = 0.0            # Reset last merged flux/volume values
                else:
                    section_params_merged = np.vstack((section_params_merged, last_merged_sections))    # Add to the merged section parameters array
                    # Re-initialise the last merged sections array with this section
                    last_merged_sections = np.array([x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, flux_section, volume_section])
                    # Re-initialise last merged flux/volume values for this section
                    last_merged_flux_section = flux_section; last_merged_volume_section = volume_section

            else:
                # Adding in this section flux does not exceed the maximum flux
                if np.isnan(last_merged_sections).any():
                    # First section in the last merged sections array
                    last_merged_sections = np.array([x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, merged_flux, merged_volume])
                else:
                    # Add to the last merged sections array
                    last_merged_sections = np.array([last_merged_sections[0],last_merged_sections[1], last_merged_sections[2],last_merged_sections[3], \
                                                x3,y3, x4,y4, last_merged_sections[8], R_section_end, merged_flux, merged_volume])
                last_merged_flux_section = merged_flux; last_merged_volume_section = merged_volume      # Update last merged flux/volume values

            # Last section. Add to the merged section parameters array if necessary
            if (sect_count + 1 > np.size(section_parameters, 0)) and not np.isnan(last_merged_sections).any():
                section_params_merged = np.vstack((section_params_merged, last_merged_sections))

    return section_params_merged

#############################################

def LargerThanBeamSize(section_coords):

    """
    Merge sections to a required number for one arm of the jet

    Parameters
    -----------
    section_coords - 1D array, shape(8,)
                     Array with section points (x,y * 4)
    
    Constants
    ---------

    Returns
    -----------
    True/False - Boolean
                 Is the section larger than the beam size

    Notes
    -----------
    """

    # Initialise return flag
    larger = False

    beam_size_pixels = RLC.beamsize / 3600 / RLC.ddel

    # Test the length of each side of the section
    if section_coords[6] == -1:
        # 3-point section
        larger = (np.sqrt( (section_coords[2]-section_coords[0])**2 + (section_coords[3]-section_coords[1])**2 ) >= beam_size_pixels) and \
                 (np.sqrt( (section_coords[4]-section_coords[2])**2 + (section_coords[5]-section_coords[3])**2 ) >= beam_size_pixels) and \
                 (np.sqrt( (section_coords[4]-section_coords[0])**2 + (section_coords[5]-section_coords[1])**2 ) >= beam_size_pixels)

    else:
        # 4-point section
        larger = (np.sqrt( (section_coords[2]-section_coords[0])**2 + (section_coords[3]-section_coords[1])**2 ) >= beam_size_pixels) and \
                 (np.sqrt( (section_coords[6]-section_coords[4])**2 + (section_coords[7]-section_coords[5])**2 ) >= beam_size_pixels) and \
                 (np.sqrt( (section_coords[4]-section_coords[2])**2 + (section_coords[5]-section_coords[3])**2 ) >= beam_size_pixels) and \
                 (np.sqrt( (section_coords[6]-section_coords[0])**2 + (section_coords[7]-section_coords[1])**2 ) >= beam_size_pixels)

    return larger

#############################################

def GetSectionParameters(area_fluxes, polygon_points, initial_polygon_points):

    """
    Get parameters for each section polygon in the jet e.g. flux, volume.

    Parameters
    -----------
    area_fluxes - 2D array,
                  raw image array

    polygon_points - 2D array, shape(n,10)
                     Array with section polygon points (x,y * 4) and distance from
                     the source.

    initial_polygon_points - 1D array, shape(5,)
                             Array with initial section polygon points (x,y * 4) and 
                             distance from the source.
    
    Constants
    ---------

    Returns
    -----------
    section_parameters - 2D array, shape(n,12)
                         Array with section polygon points (x,y * 4), distance from
                         the source and computed parameters for one arm of the jet


    Notes
    -----------
    """

    # Initialise section parameters array
    section_parameters = np.empty((0,12))

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

        # Get the section flux
        section_flux, polygon_pixel_count, overlap_pixel_count = GetFlux(area_fluxes, polypoints, last_polypoints, next_polypoints)

        # Get the section volume
        section_volume = GetVolume(polypoints)

        # Add section co-ordinates and parameters to the array
        section_parameters = np.vstack((section_parameters, 
                                        np.array([x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, section_flux, section_volume])))

        last_polypoints = np.array([x1,y1, x2,y2, x3,y3, x4,y4])    # Save the last section polygon points          

    return section_parameters

#############################################

def GetSectionPolygons(edge_points):

    """
    Returns an array of section polygon vertices and distance from the source.

    Parameters
    -----------
    edge_points - 2D array, shape(n,4)
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

    # Loop through the edge points
    point_count = 0
    for [x1, y1, x2, y2, R_section] in edge_points:
        point_count += 1

        if point_count > 1:

            # Create the array of section points
            x1_diff = np.abs(x1 - lastpts[0]); y1_diff = np.abs(y1 - lastpts[1])
            x2_diff = np.abs(x2 - lastpts[2]); y2_diff = np.abs(y2 - lastpts[3])

            # Check that there are no duplicate co-ordinates
            if x1_diff < 0.1 and y1_diff < 0.1 and x2_diff < 0.1 and y2_diff < 0.1:
                # Ignore duplicate points
                None
            else:
                # Determine whether the section has 3 or 4 vertices
                if x1_diff < 0.1 and y1_diff < 0.1:
                    current_polygon_points = np.array([lastpts[0],lastpts[1], lastpts[2],lastpts[3], x2,y2, -1,-1, last_R_section, R_section])
                elif x2_diff < 0.1 and y2_diff < 0.1:
                    current_polygon_points = np.array([lastpts[0],lastpts[1], lastpts[2],lastpts[3], x1,y1, -1,-1, last_R_section, R_section])
                else:
                    current_polygon_points = np.array([lastpts[0],lastpts[1], lastpts[2],lastpts[3], x2,y2, x1,y1, last_R_section, R_section])

                # Add section co-ordinates and parameters to the array
                polygon_points = np.vstack((polygon_points, current_polygon_points))

                last_R_section = R_section

        lastpts = np.array([x1, y1, x2, y2])

    return polygon_points

#############################################

def GetFlux(area_fluxes, curr_polypoints, last_polypoints, next_polypoints):

    """
    Returns the total flux in this section, sharing any overlap flux 
    with adjacent sections.

    Parameters
    -----------
    area_fluxes - 2D array, shape(n,2)
                  raw image array

    curr_polypoints - 1D array, shape(4,)
                      Co-ordinates of the section polygon vertices.

    last_polypoints - 1D array, shape(4,)
                      Co-ordinates of the last section polygon vertices.

    next_polypoints - 1D array, shape(4,)
                      Co-ordinates of the next section polygon vertices.
    
    Constants
    ---------

    Returns
    -----------
    section_flux - float
                   Total flux for this section of the jet

    polygon_pixel_count - integer
                          Count of pixels in the polygon.

    overlap_pixel_count - integer
                          Count of pixels in the overlap between
                          this section polygon and the last.

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
    curr_polygon_mask = polygon2mask(area_fluxes.shape, polygon_points)

    # Create the last polygon mask
    if last_polypoints[6] == -1:
        polygon_points = np.array([[last_polypoints[1],last_polypoints[0]], [last_polypoints[3],last_polypoints[2]], 
                                   [last_polypoints[5],last_polypoints[4]]])
    else:
        polygon_points = np.array([[last_polypoints[1],last_polypoints[0]], [last_polypoints[3],last_polypoints[2]], 
                                   [last_polypoints[5],last_polypoints[4]], [last_polypoints[7],last_polypoints[6]]])
    last_polygon_mask = polygon2mask(area_fluxes.shape, polygon_points)

    # Create the next polygon mask
    if not np.isnan(next_polypoints).any():         # Are we on the last section?
        if next_polypoints[6] == -1:
            polygon_points = np.array([[next_polypoints[1],next_polypoints[0]], [next_polypoints[3],next_polypoints[2]], 
                                       [next_polypoints[5],next_polypoints[4]]])
        else:
            polygon_points = np.array([[next_polypoints[1],next_polypoints[0]], [next_polypoints[3],next_polypoints[2]], 
                                       [next_polypoints[5],next_polypoints[4]], [next_polypoints[7],next_polypoints[6]]])
        next_polygon_mask = polygon2mask(area_fluxes.shape, polygon_points)
    else:
        next_polygon_mask = np.ma.make_mask(np.zeros_like(area_fluxes))

    # Make sure there is no overlap with the last and next sections
    curr_flux_mask = np.ma.mask_or(np.ma.mask_or((~curr_polygon_mask), last_polygon_mask), next_polygon_mask)

    # Get overlap masks
    last_overlap_mask = np.logical_and(curr_polygon_mask, last_polygon_mask)
    next_overlap_mask = np.logical_and(curr_polygon_mask, next_polygon_mask)

    # Determine the section and overlap pixel counts
    polygon_pixel_count = curr_polygon_mask.sum()
    overlap_pixel_count = last_overlap_mask.sum() + next_overlap_mask.sum()

    # Sum the flux in this section, sharing any overlap flux with adjacent sections
    flux_curr_polygon = np.ma.masked_array(area_fluxes, curr_flux_mask, copy = True).sum()
    if last_overlap_mask.sum() > 0: 
        flux_last_overlap = np.ma.masked_array(area_fluxes, (~last_overlap_mask), copy = True).sum()
    else: 
        flux_last_overlap = 0
    if next_overlap_mask.sum() > 0:
        flux_next_overlap = np.ma.masked_array(area_fluxes, (~next_overlap_mask), copy = True).sum()
    else:
        flux_next_overlap = 0
    section_flux = flux_curr_polygon + ((flux_last_overlap + flux_next_overlap) / 2)

    return section_flux, polygon_pixel_count, overlap_pixel_count

#############################################

def GetVolume(polypoints):

    """
    Returns the total volume of this section, by assuming the
    3D section is a cone with the vertex sliced off at an angle.
    for a 3-point section, assume the code is cut by a slant plane
    so that the cut just touches the base.

    Parameters
    -----------
    polypoints - 1D array, shape(4,)
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

        # Calculate the volume
        cone_R = base_points[1,0] / 2                                           # Radius of cone base.
        cone_cotPhi = top_point[1] / top_point[0]                               # cot of cone half-angle.
        cone_H = cone_R * cone_cotPhi                                           # Height of cone.
        cone_slant_m = top_point[1] / (base_points[1,0] - top_point[0])         # Gradient of slant plane.
        cone_tanTheta = np.abs(cone_slant_m)                                    # Angle from horizontal of slant plane.

        # Volume of top of the cone, from the vertex down to the slant plane
        cone_volume_top = pi/3 * cone_R**2 * cone_H * \
                          pow( ((cone_H - (cone_tanTheta * cone_R)) / (cone_H + (cone_tanTheta * cone_R))), 3/2)

        # Total volume of the cone
        cone_volume = pi/3 * cone_R**2 * cone_H

        # Volume of the base of the cone, from the base up to the slant plane
        cone_volume_base = cone_volume - cone_volume_top

        section_volume = cone_volume_base

    else:
        # 4 points in polygon
        base_points, top_points = Setup4PointPolygon(polypoints)

        # Calculate the volume
        cone_R = base_points[1,0] / 2                                                               # Radius of cone base.
        cone_cotPhi = top_points[0,1] / top_points[0,0]                                             # cot of cone half-angle.
        cone_H = cone_R * cone_cotPhi                                                               # Height of cone.
        cone_slant_m = (top_points[1,1] - top_points[0,1]) / (top_points[1,0] - top_points[0,0])    # Gradient of slant plane.
        cone_tanTheta = np.abs(cone_slant_m)                                                        # Angle from horizontal of slant plane.
        cone_h1 = ( cone_slant_m * (base_points[1,0] / 2) ) + top_points[0,1]                       # Distance of slant plane above
                                                                                                    # the base, along the cone axis.
        cone_h = cone_H - cone_h1                                                                   # Distance of slant plane from the
                                                                                                    # vertex, along the cone axis.
        # Volume of top of the cone, from the vertex down to the slant plane
        cone_volume_top = (pi/3 * pow(cone_h,3) * cone_cotPhi) / pow( (cone_cotPhi**2 - cone_tanTheta**2), 3/2)

        # Total volume of the cone
        cone_volume = pi/3 * cone_R**2 * cone_H

        # Volume of the base of the cone, from the base up to the slant plane
        cone_volume_base = cone_volume - cone_volume_top

        section_volume = cone_volume_base

    return section_volume

#############################################

def Setup3PointPolygon(polypoints):

    """
    Sets up a 3-point section polygon, ready for the volume calculation.

    Parameters
    -----------
    polypoints - 1D array, shape(8,)
                 Co-ordinates of the section polygon vertices.
    
    Constants
    ---------

    Returns
    -----------
    base_points - 1D array, shape(4,)
                  Co-ordinates of the polygon base

    top_point - 1D array, shape(2,)
                Co-ordinate of the polygon top

    Notes
    -----------
    """

    # Determine the base and top co-ordinate of the polygon "partial cone".
    length1 = np.sqrt( (polypoints[2]-polypoints[0])**2 + (polypoints[3]-polypoints[1])**2 )
    length2 = np.sqrt( (polypoints[4]-polypoints[2])**2 + (polypoints[5]-polypoints[3])**2 )
    length3 = np.sqrt( (polypoints[4]-polypoints[0])**2 + (polypoints[5]-polypoints[1])**2 )
    if length1 > length2 and length1 > length3:
        base_points = np.array([[polypoints[0], polypoints[1]], [polypoints[2], polypoints[3]]])        # Base co-ordinates
        top_point = np.array([polypoints[4], polypoints[5]])                                            # Top co-ordinate
    elif length2 > length1 and length2 > length3:
        base_points = np.array([[polypoints[2], polypoints[3]], [polypoints[4], polypoints[5]]])        # Base co-ordinates
        top_point = np.array([polypoints[0], polypoints[1]])                                            # Top co-ordinate
    else:
        base_points = np.array([[polypoints[4], polypoints[5]], [polypoints[0], polypoints[1]]])        # Base co-ordinates
        top_point = np.array([polypoints[2], polypoints[3]])                                            # Top co-ordinate

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
    polypoints - 1D array, shape(8,)
                 Co-ordinates of the section polygon vertices.
    
    Constants
    ---------

    Returns
    -----------
    base_points - 1D array, shape(4,)
                  Co-ordinates of the polygon base

    top_point - 1D array, shape(4,)
                Co-ordinates of the polygon top

    Notes
    -----------
    """

    # Determine the base and top co-ordinates of the polygon "partial cone".
    length1 = np.sqrt( (polypoints[2] - polypoints[0])**2 + (polypoints[3] - polypoints[1])**2 )
    length2 = np.sqrt( (polypoints[6] - polypoints[4])**2 + (polypoints[7] - polypoints[5])**2 )
    if length1 > length2:                                                                               # Compare possible base/top lengths
        base_points = np.array([[polypoints[0], polypoints[1]], [polypoints[2], polypoints[3]]])        # Base co-ordinates
        top_points = np.array([[polypoints[4], polypoints[5]], [polypoints[6], polypoints[7]]])         # Top co-ordinates
    else:
        base_points = np.array([[polypoints[4], polypoints[5]], [polypoints[6], polypoints[7]]])        # Base co-ordinates
        top_points = np.array([[polypoints[0], polypoints[1]], [polypoints[2], polypoints[3]]])         # Top co-ordinates

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
    if rotated_shape[2,0] < rotated_shape[3,0]:                     # Order top points in order of increasing X
        top_points = np.array([[rotated_shape[2,0],rotated_shape[2,1]], [rotated_shape[3,0],rotated_shape[3,1]]])
    else:
        top_points = np.array([[rotated_shape[3,0],rotated_shape[3,1]], [rotated_shape[2,0],rotated_shape[2,1]]])
    base_points[0,0] = 0.0; base_points[0,1] = 0.0                  # Ensure first base point is exactly at the origin
    base_points[1,1] = 0.0                                          # Ensure second base point lies exactly on the X axis

    # If the top points have -ve Y co-ordinates, reflect around the X axis
    if top_points[0,1] < 0:
        top_points[0,1] = - top_points[0,1]; top_points[1,1] = - top_points[1,1]

    # If the gradient of the first side is negative or the gradient of the second side is positive,
    # move the X co-ordinate such that the gradient is negated. This may happen for the short side
    # of the polygon, when at a bend in the jet.
    if (top_points[0,0] - base_points[0,0]) < 0:
        top_points[0,0] += np.abs(top_points[0,0]-base_points[0,0]) * 2
    if (top_points[1,0] - base_points[1,0]) > 0:
        top_points[1,0] -= np.abs(top_points[1,0]-base_points[1,0]) * 2

    # Make an approximation. Move the top co-ordinates such that the gradient of the shortest
    # side is the same (although opposite) as that of the longest side.
    length_side1 = np.sqrt( (top_points[0,0]-base_points[0,0])**2 + (top_points[0,1]-base_points[0,1])**2 )
    length_side2 = np.sqrt( (top_points[1,0]-base_points[1,0])**2 + (top_points[1,1]-base_points[1,1])**2 )
    grad_side1 = (top_points[0,1] - base_points[0,1]) / (top_points[0,0] - base_points[0,0])
    grad_side2 = (top_points[1,1] - base_points[1,1]) / (top_points[1,0] - base_points[1,0])
    if length_side1 < length_side2:
        base_points[0,0] = top_points[0,0] - ( (top_points[0,1]-base_points[0,1]) / (- grad_side2) )
        offset = np.array([base_points[0,0], base_points[0,1]])
        base_points -= offset; top_points -= offset 
        base_points[0,0] = 0.0; base_points[0,1] = 0.0              # Ensure first base point is exactly at the origin
        base_points[1,1] = 0.0                                      # Ensure second base point lies exactly on the X axis
    else:
        base_points[1,0] = top_points[1,0] - ( (top_points[1,1]-base_points[1,1]) / (- grad_side1) )

    # If the sides of the polygon are less than 1 degree from the vertical, move the   
    # top co-ordinates to make the sides 1 degree from the vertical
    if atan2(top_points[0,0], top_points[0,1]) < pi*1/180:
        top_points[0,0] = top_points[0,1] * tan(pi*1/180)
        top_points[1,0] = base_points[1,0] - (top_points[1,1] * tan(pi*1/180))

    return base_points, top_points

#############################################

def SaveEdgepointFiles(source_name, edge_points1, edge_points2, section_parameters1, section_parameters2):

    """
    Saves the edge point file for each arm of the jet

    Parameters
    -----------
    source_name - str,
                  the name of the source

    edge_points1 - 2D array, shape(n,5)
                   Points on one arm of the jet, corresponding to the 
                   closest distance to that edge from the corresponding ridge point,
                   and their distance from source.

    edge_points2 - 2D array, shape(n,5)
                   Points on the other arm of the jet, corresponding to the 
                   closest distance to that edge from the corresponding ridge point,
                   and their distance from source.

    section_parameters1 - 2D array, shape(n,12)
                          Array for one arm of the jet, with section points (x/y * 4), 
                          distance from source and computed parameters

    section_parameters2 - 2D array, shape(n,12)
                          Array for other arm of the jet, with section points (x/y * 4), 
                          distance from source and computed parameters
    
    Constants
    ---------

    Returns
    -----------

    Notes
    -----------
    """

    try:
        fileEP1 = np.column_stack((edge_points1[:,0], edge_points1[:,1], edge_points1[:,2], edge_points1[:,3], edge_points1[:,4]))
        fileEP2 = np.column_stack((edge_points2[:,0], edge_points2[:,1], edge_points2[:,2], edge_points2[:,3], edge_points2[:,4]))
        np.savetxt(RLF.EP1 %source_name, fileEP1, delimiter=' ')
        np.savetxt(RLF.EP2 %source_name, fileEP2, delimiter=' ')
    except Exception as e:
        print('Error occurred saving edgepoint files')

    try:
        fileSP1 = np.column_stack((section_parameters1[:,0], section_parameters1[:,1], section_parameters1[:,2], section_parameters1[:,3], \
                                   section_parameters1[:,4], section_parameters1[:,5], section_parameters1[:,6], section_parameters1[:,7], \
                                   section_parameters1[:,8], section_parameters1[:,9], section_parameters1[:,10], section_parameters1[:,11]))
        fileSP2 = np.column_stack((section_parameters2[:,0], section_parameters2[:,1], section_parameters2[:,2], section_parameters2[:,3], \
                                   section_parameters2[:,4], section_parameters2[:,5], section_parameters2[:,6], section_parameters2[:,7], \
                                   section_parameters2[:,8], section_parameters2[:,9], section_parameters2[:,10], section_parameters2[:,11]))
        np.savetxt(RLF.SP1 %source_name, fileSP1, delimiter=' ')
        np.savetxt(RLF.SP2 %source_name, fileSP2, delimiter=' ')
    except Exception as e:
        print('Error occurred saving section parameters files')

#############################################

def PlotEdgePoints(area_fluxes, source_name, dphi, edge_points1, edge_points2, section_parameters1, section_parameters2):

    """
    Plots the edge points on the source.
    Plots the jet sections on the source.

    Parameters
    -----------
    area_fluxes - 2D array,
                  raw image array

    source_name - str,
                  the name of the source

    dphi - float,
           1/2 of the value of the ridgepoint cone opening angle

    edge_points1 - 2D array, shape(n,4)
                   Points on one arm of the jet, corresponding to the 
                   closest distance to that edge from the corresponding ridge point.

    edge_points2 - 2D array, shape(n,4)
                   Points on the other arm of the jet, corresponding to the 
                   closest distance to that edge from the corresponding ridge point.

    section_parameters1 - 2D array, shape(n,12)
                          Array for one arm of the jet, with section points (x/y * 4), 
                          distance from source and computed parameters

    section_parameters2 - 2D array, shape(n,12)
                          Array for other arm of the jet, with section points (x/y * 4), 
                          distance from source and computed parameters
    
    Constants
    ---------

    Returns
    -----------

    Notes
    -----------
    """

    try:
        palette = plt.cm.cividis
        palette = copy.copy(plt.cm.get_cmap("cividis"))
        palette.set_bad('k',0.0)
        lmsize = RLG.sSize  # pixels
        optical_pos = (float(lmsize), float(lmsize))

        y, x = np.mgrid[slice((0),(area_fluxes.shape[0]),1), slice((0),(area_fluxes.shape[1]),1)]
        y = np.ma.masked_array(y, mask=np.ma.masked_invalid(area_fluxes).mask)
        x = np.ma.masked_array(x, mask=np.ma.masked_invalid(area_fluxes).mask)
                        
        xmin = np.ma.min(x)
        xmax = np.ma.max(x)
        ymin = np.ma.min(y)
        ymax = np.ma.max(y)
                        
        x_source_min = float(optical_pos[0]) - RLC.ImSize * float(lmsize)
        x_source_max = float(optical_pos[0]) + RLC.ImSize * float(lmsize)
        y_source_min = float(optical_pos[1]) - RLC.ImSize * float(lmsize)
        y_source_max = float(optical_pos[1]) + RLC.ImSize * float(lmsize)
                        
        if x_source_min < xmin:
            xplotmin = xmin
        else:
            xplotmin = x_source_min
                                
        if x_source_max < xmax:
            xplotmax = x_source_max
        else:
            xplotmax = xmax
                        
        if y_source_min < ymin:
            yplotmin = ymin
        else:
            yplotmin = y_source_min
                                
        if y_source_max < ymax:
            yplotmax = y_source_max
        else:
            yplotmax = ymax

        # Plot edge points
        fig, ax = plt.subplots(figsize=(10,10))
        fig.suptitle('Source: %s' %source_name)
        fig.subplots_adjust(top=0.9)
        ax.set_aspect('equal', 'datalim')
    
        A = np.ma.array(area_fluxes, mask=np.ma.masked_invalid(area_fluxes).mask)
        ax.pcolor(x, y, A, cmap=palette, vmin=np.nanmin(A), vmax=np.nanmax(A))
        ax.plot(edge_points1[:,0], edge_points1[:,1], 'r-', linewidth=0.6) # Edge lines
        ax.plot(edge_points1[:,2], edge_points1[:,3], 'r-', linewidth=0.6)
        ax.plot(edge_points2[:,0], edge_points2[:,1], 'r-', linewidth=0.6)
        ax.plot(edge_points2[:,2], edge_points2[:,3], 'r-', linewidth=0.6)
        for ep in edge_points1:
            x_values = np.array([ep[0], ep[2]])
            y_values = np.array([ep[1], ep[3]])
            ax.plot(x_values, y_values, 'y-', linewidth=0.6)               # Edge point segment separators
        for ep in edge_points2:
            x_values = np.array([ep[0], ep[2]])
            y_values = np.array([ep[1], ep[3]])
            ax.plot(x_values, y_values, 'y-', linewidth=0.6)               # Edge point segment separators
        ax.legend()
        ax.set_xlim(xplotmin, xplotmax)
        ax.set_ylim(yplotmin, yplotmax)
    
        fig.savefig(RLF.EPimage %(source_name, dphi))
        plt.close(fig)

        # Plot jet sections
        fig, ax = plt.subplots(figsize=(10,10))
        fig.suptitle('Source: %s' %source_name)
        fig.subplots_adjust(top=0.9)
        ax.set_aspect('equal', 'datalim')
    
        A = np.ma.array(area_fluxes, mask=np.ma.masked_invalid(area_fluxes).mask)
        ax.pcolor(x, y, A, cmap=palette, vmin=np.nanmin(A), vmax=np.nanmax(A))
        for ep in section_parameters1:
            x_values = np.array([ep[0], ep[2]])
            y_values = np.array([ep[1], ep[3]])
            ax.plot(x_values, y_values, 'y-', linewidth=0.6)               # Segment separators
        for ep in section_parameters2:
            x_values = np.array([ep[0], ep[2]])
            y_values = np.array([ep[1], ep[3]])
            ax.plot(x_values, y_values, 'y-', linewidth=0.6)               # Segment separators
        ax.legend()
        ax.set_xlim(xplotmin, xplotmax)
        ax.set_ylim(yplotmin, yplotmax)
    
        fig.savefig(RLF.SCimage %(source_name, dphi))
        plt.close(fig)
    except:
        print('Error occurred plotting edgepoints')

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

    pos - 1D array, shape(2,)
          array containing central point position given as pixel
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

    if 0 <= angle <= pi/2.:
        quadrant = 1

    elif pi/2. <= angle <= pi:
        quadrant = 2

    elif -pi <= angle <= -pi/2.:
        quadrant = 3

    elif -pi/2. <= angle <= 0:
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