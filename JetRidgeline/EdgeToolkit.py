# EdgeToolkit.py
# Toolkit for edge-point finding in a source
# Created by Liz - Jan 2025

from ast import While
import JetRidgeline.RidgelineFiles as RLF
import JetRidgeline.RLConstants as RLC
import JetRidgeline.RLGlobal as RLG
from JetRidgeline.RidgeToolkit import CheckQuadrant, PiRange, PolarCoordinates
from math import atan2
from numpy import pi
import numpy as np

#############################################

def FindEdgePoints(area_fluxes, ridge_point, ridge_phi, prev_edge_points):

    """
    Returns edge points, on either side of the jet, corresponding to
    the closest distance to that edge from the supplied ridge point.

    Parameters
    -----------
    area_fluxes - 2D array,
                  raw image array

    ridge_point - 1D array, shape(2, )
                  ridge point along the ridge line

    ridge_phi - float,
                angular direction of the ridge line

    prev_edge_points - 1D array, shape(4,)
                       edge points corresponding to
                       previous ridge point
                       (x1,y1,x2,y2)
    
    Constants
    ---------

    Returns
    -----------

    stop_finding_edge_points - flag for no further edge points to be located

    edge_points - 1D array, shape(4,)
                  Points on either side of the jet, corresponding to the 
                  closest distance to that edge from the corresonding ridge point.
                  (x1,y1,x2,y2)

    Notes
    -----------

    """

    # Initialise flag to indicate that edge points determination should stop
    stop_finding_edge_points = False

    # Initialise edge_points array
    edge_points = np.full(4, np.nan)

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
    jet_mask = np.ma.masked_where(area_fluxes_valid > (RLC.nSig * 0.0002), area_fluxes_valid).mask

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
    edge_coord1 = np.array([edge_coord1_yx[1],edge_coord1_yx[0]])

    # If the ridge point is very close to an edge of the jet, the second edge point
    # could be found on the same side of the jet. Reduce the angle of search until the
    # second edge point is found on the other side of the jet.

    edge_coord2 = np.full(2, np.nan)        # Initialise 2nd edge point array
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
            edge_coord2 = np.array([edge_coord2_yx[1],edge_coord2_yx[0]])

    if not np.isnan(edge_coord2).any():
        # Detemine which should be the first co-ordinate in the array.
        if phi_prev1 == phi_prev_coord1:
            edge_points = np.concatenate((edge_coord1, edge_coord2), axis=0)
        else:
            edge_points = np.concatenate((edge_coord2, edge_coord1), axis=0)

    return stop_finding_edge_points, edge_points

#############################################

def FindInitEdgePoints(area_fluxes, ridge_point, ridge_phi):

    """
    Returns edge points, on either side of the jet, corresponding to
    the closest distance to that edge from the initial ridge point.

    Parameters
    -----------
    area_fluxes - 2D array,
                  raw image array

    ridge_point - 1D array, shape(2, )
                  ridge point along the ridge line

    ridge_phi - float,
                angular direction of the ridge line
    
    Constants
    ---------

    Returns
    -----------

    edge_points - 1D array, shape(4,)
                  Points on either side of the jet, corresponding to the 
                  closest distance to that edge from the initial ridge point.
                  (x1,y1,x2,y2)

    Notes
    -----------

    """

    # Initialise edge_points array
    edge_points = np.full(4, np.nan)

    # Get polar coordinates for area_fluxes, around the ridge point
    r, phi = PolarCoordinates(area_fluxes, ridge_point)

    # Fill invalid (nan) values with zeroes in area_fluxes
    area_fluxes_valid = np.ma.filled(np.ma.masked_invalid(area_fluxes), 0)

    # Mask the jet - where flux values are above (RLC.nSig * rms)
    jet_mask = np.ma.masked_where(area_fluxes_valid > (RLC.nSig * 0.0002), area_fluxes_valid).mask

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
    edge_coord1 = np.array([edge_coord1_yx[1],edge_coord1_yx[0]])

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
    edge_coord2 = np.array([edge_coord2_yx[1],edge_coord2_yx[0]])

    # Add to the edge points array in order of phi value
    if phi[edge_coord1_yx] < phi[edge_coord2_yx]:
        edge_points = np.concatenate((edge_coord1, edge_coord2), axis=0)
    else:
        edge_points = np.concatenate((edge_coord2, edge_coord1), axis=0)

    return edge_points

#############################################
