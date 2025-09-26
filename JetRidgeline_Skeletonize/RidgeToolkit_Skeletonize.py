#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RidgeToolkit_Skeleton.py
Toolkit for creating a ridgeline via 
skimage skeletonize, as developed by Hugh Dickenson,
then transforming the data into the form required 
for edgepoint and section creation.
Created by LizWhitehead - 24/08/2025
"""

import JetModelling_MapSetup as JMS
import JetModelling_SourceSetup as JSS
import JetModelling_Constants as JMC
import JetModelling_MapAnalysis as JMA
import JetRidgeline_Skeletonize.RidgelineFiles_Skeletonize as RLSF
from skan.csr import make_degree_image
import skimage as ski
from distributed import Client
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from math import atan2, floor, nan, isnan
import copy

MaxContourLength = 2000     # Maximum allowed length for contours

#############################################

def CreateAndLoadSkeletonRidgeline(flux_array):
    """
    Create and load ridgeline data files.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    Returns
    -----------
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
    """
    ridge_coords1, ridge_coords2 = CreateSkeletonRidgeline(flux_array)
    ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2 = LoadRidgelineData(flux_array, ridge_coords1, ridge_coords2)

    return ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2

#############################################

def CreateSkeletonRidgeline(flux_array):
    """
    Create ridgeline data files by using skimage skeletonize, 
    as developed by Hugh Dickenson, to create the ridgeline.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    Returns
    -----------
    ridge_coords1 - 2D array, shape(n,2)
                    Array of ridgepoint co-ordinates for one arm of the jet
    
    ridge_coords2 - 2D array, shape(n,2)
                    Array of ridgepoint co-ordinates for other arm of the jet
    """
    cutdown_amount0 = floor( (flux_array.shape[0] - JMS.CutdownSize0) / 2.0 )
    cutdown_amount1 = floor( (flux_array.shape[1] - JMS.CutDownSize1) / 2.0 )
    image = flux_array[cutdown_amount0 : (flux_array.shape[0]-cutdown_amount0), \
                       cutdown_amount1 : (flux_array.shape[1]-cutdown_amount1),]
    image_orig = image.copy()

    coords, image_brightest = GetSkeleton(image_orig, image, JMS.nSig_s)
    coords1, coords2 = split_at_brightest(coords.T.tolist(), image_brightest)
    coords1 = RemoveLoopingEnds(coords1.tolist())
    coords2 = RemoveLoopingEnds(coords2.tolist())

    if JMS.SplitInnerOuterSkeleton:
        coords_outer, image_brightest = GetSkeleton(image_orig, image, JMS.nSig_s_outer)
        coords_outer1, coords_outer2 = split_at_brightest(coords_outer.T.tolist(), image_brightest)
        coords_outer1 = RemoveLoopingEnds(coords_outer1.tolist())
        coords_outer2 = RemoveLoopingEnds(coords_outer2.tolist())
        coords1 = CombineInnerAndOuterSkeletons(coords1.tolist(), coords_outer1.tolist())
        coords2 = CombineInnerAndOuterSkeletons(coords2.tolist(), coords_outer2.tolist())

    coords1[:,0] += cutdown_amount0; coords1[:,1] += cutdown_amount1
    coords2[:,0] += cutdown_amount0; coords2[:,1] += cutdown_amount1

    ridge_coords1 = np.column_stack((coords1[:,1], coords1[:,0]))
    ridge_coords2 = np.column_stack((coords2[:,1], coords2[:,0]))

    return ridge_coords1, ridge_coords2

######################################################################

def GetSkeleton(image_orig, image, nSig):
    """
    Create a skeleton and order the points in it.

    Parameters
    -----------
    image_orig - 2D array,
                 Original raw image array

    image - 2D array,
            Cutdown image array

    nsig - float
           The multiple of sigma for RMS reduction

    Returns
    -----------
    coords - 2D array, shape(n,2)
             Array of ordered skeleton co-ordinates 
    
    image_brightest - 1D array
                      Coordinates of the brightest point in the masked image
    """
    if not isnan(nSig):
        image_valid = np.ma.filled(np.ma.masked_invalid(image), 0)
        rms_mask = np.ma.masked_where(image_valid <= (nSig * JMS.bgRMS), image_valid).mask
        image = np.ma.masked_array(image, mask = rms_mask)

    image = ski.filters.gaussian(image, sigma=JMS.GaussSigmaFilter)
    contours = ski.measure.find_contours(image, np.percentile(image, JMS.ContoursLevelPerc))
    max_pos = np.argmax([len(contour) for contour in contours])

    mask_image = np.zeros(image.T.shape)
    mask_image[ski.draw.polygon(*contours[max_pos].T)] = 1
    if not isnan(nSig): mask_image[rms_mask] = 0

    image = image_orig.copy()
    image[~(mask_image > 0)] = 0

    image_brightest = np.unravel_index(np.nanargmax(image), image.shape)

    thresholds = 100 - np.logspace(-3, 2, 25)
    percentiles = np.percentile(image, thresholds)

    with Client() as client:
        remote_image = client.scatter(image)

        futures = client.map(
            process_level,
            percentiles[percentiles > 0],
            [remote_image] * len(percentiles[percentiles > 0]),
        )
        skeletons = client.gather(futures)

    skeleton_sum = np.sum([s for s in skeletons if s is not None], axis=0)

    thresholded = skeleton_sum > 0
    thresholded = ski.morphology.skeletonize(
        ski.morphology.remove_small_holes(ski.filters.gaussian(thresholded, 1) > 0, JMS.MaxRemoveSmallHolesArea)
    )

    num_ends = np.inf
    while num_ends > 2:
        degrees = make_degree_image(thresholded)
        ends = degrees == 1
        num_ends = ends.sum()
        thresholded[ends] = 0
        thresholded = ski.morphology.skeletonize(thresholded)

    coords = np.array(np.nonzero(thresholded))

    ends = np.array(np.nonzero(make_degree_image(thresholded) == 1))

    end_indices = [
        next((idx for idx, val in enumerate(coords.T) if np.all(val == end)), None)
        for end in ends.T
    ]

    coords = order_points(coords.T.tolist(), end_indices[1])[:-3].T

    return coords, image_brightest

######################################################################

def RemoveLoopingEnds(coords):
    """
    Remove the end section of the coordinates where the skeleton turns back on itself

    Parameters
    -----------
    coords - 2D array, shape(n,2)
             Array of ordered skeleton co-ordinates 

    Returns
    -----------
    coords_updated - 2D array, shape(n,2)
                     Array of ordered skeleton co-ordinates with the "loop" removed
    
    """
    coords_updated = []

    icnt = 0
    for coord in coords:
        if icnt > 0:
            dist = np.sqrt( (coord[0]-prev_coord[0])**2 + (coord[1]-prev_coord[1])**2 )
            if dist > JMS.MaximumLoopJumpPixels:
                break

        coords_updated.append(coord)

        prev_coord = coord
        icnt += 1

    return np.array(coords_updated)

######################################################################

def CombineInnerAndOuterSkeletons(coords_inner, coords_outer):
    """
    Combine skeletons by joining the end of the inner skeleton with the
    nearest point on the outer skeleton, keeping the outer points of the
    outer skeleton.

    Parameters
    -----------
    coords_inner - 2D array, shape(n,2)
                   Array of ordered inner skeleton co-ordinates 

    coords_outer - 2D array, shape(n,2)
                   Array of ordered outer skeleton co-ordinates 

    Returns
    -----------
    coords - 2D array, shape(n,2)
             Array of ordered skeleton co-ordinates with the "loop" removed
    
    """
    join_pt_inner = coords_inner[len(coords_inner)-1]
    dist = np.linalg.norm(np.array(coords_outer) - np.array(join_pt_inner), axis=1)
    join_ind_outer = dist.argmin()

    # Attempt to smooth the join by taking points before the join points and interpolating
    join_ind_inner = len(coords_inner) - floor(JMS.JoinInterpolatePoints/2) - 1; join_pt_inner = coords_inner[join_ind_inner]
    join_ind_outer += floor(JMS.JoinInterpolatePoints/2); join_pt_outer = coords_outer[join_ind_outer]
    intCnt = 1; interpolate_pts = []
    while intCnt <= JMS.JoinInterpolatePoints:
        interpolate_pts.append([join_pt_inner[0] + ((join_pt_outer[0] - join_pt_inner[0]) * intCnt / (JMS.JoinInterpolatePoints+1)), \
                                join_pt_inner[1] + ((join_pt_outer[1] - join_pt_inner[1]) * intCnt / (JMS.JoinInterpolatePoints+1))])
        intCnt += 1

    coords = coords_inner[:join_ind_inner+1] + interpolate_pts + coords_outer[join_ind_outer:]

    return np.array(coords)

######################################################################

def split_at_brightest(points, image_brightest):
    """
    Split skeleton coordinates at the brightest point.

    Parameters
    -----------
    points - 2D array, shape(n,2)
             Array of ordered skeleton co-ordinates 

    image_brightest - 1D array
                      Coordinates of the brightest point in the masked image

    Returns
    -----------
    pointsArm1 - 2D array, shape(n,2)
                 Array of ordered skeleton co-ordinates in first split
    
    pointsArm2 - 2D array, shape(n,2)
                 Array of ordered skeleton co-ordinates in second split
    """
    # Find the skleton point closest to the brightest point in the image
    pbright = image_brightest
    d = np.linalg.norm(np.array(points) - np.array(pbright), axis=1)
    ind = d.argmin()

    # Split such that the coordinates with the highest Y value are in the first split
    if points[0][0] > points[ind][0]:
        pointsArm2 = points[ind:]
        pointsOtherArm = points[:ind+1]
        pointsArm1 = []
        for point in reversed(pointsOtherArm):
            pointsArm1.append(point)
    else:
        pointsArm1 = points[ind:]
        pointsOtherArm = points[:ind]
        pointsArm2 = []
        for point in reversed(pointsOtherArm):
            pointsArm2.append(point)

    return np.array(pointsArm1), np.array(pointsArm2)

######################################################################

def order_points(points, ind):
    """
    Order the points in the skeleton.

    Parameters
    -----------
    points - 2D array, shape(n,2)
             Array of un-ordered skeleton co-ordinates 

    ind - integer
          Index of point at one end of the skeleton

    Returns
    -----------
    points_new - 2D array, shape(n,2)
                 Array of ordered skeleton co-ordinates
    """
    points_new = [
        points.pop(ind)
    ]                                       # initialize a new list of points with the known first point
    pcurr = points_new[-1]                  # initialize the current point (as the known point)
    while len(points) > 0:
        d = np.linalg.norm(
            np.array(points) - np.array(pcurr), axis=1
        )                                   # distances between pcurr and all other remaining points
        ind = d.argmin()                    # index of the closest point
        points_new.append(points.pop(ind))  # append the closest point to points_new
        pcurr = points_new[-1]              # update the current point

    return np.array(points_new)

######################################################################

def process_level(level, image):
    """
    Function run in parallel processes.

    Parameters
    -----------
    level - Array of percentiles of image values 

    image - 2D array,
            Partial masked image array


    Returns
    -----------
    skeleton - 2D array, shape(n,2)
               Array of partial skeleton co-ordinates
    """
    contours = ski.measure.find_contours(image, level)

    if len(contours) > MaxContourLength or len(contours) < 1:
        return None

    longest_contour = contours[np.argmax([len(contour) for contour in contours])]

    if len(longest_contour) > 50:
        contour_image = np.zeros(image.T.shape)
        contour_image[ski.draw.polygon(*longest_contour.T)] = 1
        skeleton = ski.morphology.skeletonize(contour_image)

        num_ends = np.inf

        while num_ends > 2:
            degrees = make_degree_image(skeleton)
            ends = degrees == 1
            num_ends = ends.sum()
            skeleton[ends] = 0
            skeleton = ski.morphology.skeletonize(skeleton)

        return skeleton
    
#############################################

def LoadRidgelineData(flux_array, ridge_coords1, ridge_coords2):
    """
    Load existing ridgeline data into internal data
    structures, required for edgepoint and section creation.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    Returns
    -----------
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
    """
    # Read ridgeline data file for first arm
    ridge1 = np.empty((0,2)); phi_val1 = np.empty((0)); Rlen1 = np.empty((0))
    ridge1, phi_val1, Rlen1 = LoadRidgelineDataForOneArm(ridge_coords1, ridge1, phi_val1, Rlen1)

    # Read ridgeline data file for other arm
    ridge2 = np.empty((0,2)); phi_val2 = np.empty((0)); Rlen2 = np.empty((0))
    ridge2, phi_val2, Rlen2 = LoadRidgelineDataForOneArm(ridge_coords2, ridge2, phi_val2, Rlen2)

    # Determine the source position and update all data to be relative to this position
    sCentre, ridge1, ridge2, Rlen1, Rlen2, phi_val1, phi_val2 = \
        SetDataRelativeToSourcePosition(flux_array, ridge1, ridge2, Rlen1, Rlen2, phi_val1, phi_val2)

    # Save files and plot data
    SaveRidgelineFiles(ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2)
    PlotRidgelines(flux_array, sCentre, ridge1, ridge2)

    return ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2

#############################################

def LoadRidgelineDataForOneArm(ridge_coords, ridge, phi_val, Rlen):

    """
    Load existing ridgeline data into internal data
    structures, for one arm of the jet

    Parameters
    -----------
    input_file_name - string
                      Name of input text file

    ridge - 2D array, shape(n,2)
            Array of ridgepoint co-ordinates for one arm of the jet

    phi_val - 1D array of ridgeline angles for each ridgepoint on
              one arm of the jet

    Rlen - 1D array of distance from source for each ridgepoint on
           one arm of the jet

    Returns
    -----------
    ridge - 2D array, shape(n,2)
            Array of ridgepoint co-ordinates for one arm of the jet

    phi_val - 1D array of ridgeline angles for each ridgepoint on
              one arm of the jet

    Rlen - 1D array of distance from source for each ridgepoint on
           one arm of the jet
    """

    # Read all data from the input file
    lineidx = 0; latest_R = 0.0
    for [x, y] in ridge_coords:
        if lineidx == 0:
            x = float(x); y = float(y)
            ridge = np.vstack((ridge, np.array([x, y])))
            Rlen = np.hstack((Rlen, 0.0))
            last_saved_x = x; last_saved_y = y; last_saved_R = 0.0
        else:
            x = float(x); y = float(y)
            latest_R += np.sqrt( (x - last_saved_x)**2 + (y - last_saved_y)**2 )
            if (latest_R - last_saved_R) > JMC.R_s:                 # Only save when difference in R > defined max value
                ridge = np.vstack((ridge, np.array([x, y])))
                Rlen = np.hstack((Rlen, latest_R))
                phi_val = np.hstack(( phi_val, atan2((y - last_saved_y), (x - last_saved_x)) ))
                last_saved_x = x; last_saved_y = y; last_saved_R = latest_R

        lineidx += 1

    # Save the last data if necessary
    if x != last_saved_x or y != last_saved_y:
        ridge = np.vstack((ridge, np.array([x, y])))
        Rlen = np.hstack((Rlen, latest_R))
        phi_val = np.hstack(( phi_val, atan2((y - last_saved_y), (x - last_saved_x)) ))

    # Add a final phi value, the same as the previous one
    phi_val = np.hstack((phi_val, phi_val[-1]))

    # Add a null row
    ridge = np.vstack((ridge, np.array([np.nan, np.nan])))
    Rlen = np.hstack((Rlen, np.nan))
    phi_val = np.hstack((phi_val, np.nan))

    return ridge, phi_val, Rlen

#############################################

def SaveRidgelineFiles(ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2):

    """
    Saves the ridgeline files for each arm of the jet

    Parameters
    -----------
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
    """

    file1 = np.column_stack((ridge1[:,0], ridge1[:,1], phi_val1, Rlen1))
    file2 = np.column_stack((ridge2[:,0], ridge2[:,1], phi_val2, Rlen2))
    np.savetxt(RLSF.R1 %(JMS.sName, str(JMS.map_number+1)), file1, delimiter=' ', \
               header='ridgepoint x-coord (pix), ridgepoint y-coord (pix), ridgepoint phi (radns), ridgepoint R (pix)')
    np.savetxt(RLSF.R2 %(JMS.sName, str(JMS.map_number+1)), file2, delimiter=' ', \
               header='ridgepoint x-coord (pix), ridgepoint y-coord (pix), ridgepoint phi (radns), ridgepoint R (pix)')

#############################################

def PlotRidgelines(flux_array, sCentre, ridge1, ridge2):

    """
    Plots the edge points and jet sections on the source.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    sCentre - 1D array, shape(2,)
              Pixel co-ordinates of the source centre

    ridge1 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for one arm of the jet
    
    ridge2 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for other arm of the jet

    """

    flux_array_plot = JMC.flux_factor * flux_array.copy()

    palette = plt.cm.cividis
    palette = copy.copy(plt.get_cmap("cividis"))
    palette.set_bad('k',0.0)
    lmsize = JMS.sSize  # pixels

    y, x = np.mgrid[slice((0),(flux_array_plot.shape[0]),1), slice((0),(flux_array_plot.shape[1]),1)]
    y = np.ma.masked_array(y, mask=np.ma.masked_invalid(flux_array_plot).mask)
    x = np.ma.masked_array(x, mask=np.ma.masked_invalid(flux_array_plot).mask)

    y_plotlimits = np.ma.masked_array(y, mask=np.ma.masked_where(y < (np.min(JMC.nSig_arms) * JMS.bgRMS), y, copy=True).mask)
    x_plotlimits = np.ma.masked_array(x, np.ma.masked_where(x < (np.min(JMC.nSig_arms) * JMS.bgRMS), x, copy=True).mask)
    xmin = np.ma.min(x_plotlimits)
    xmax = np.ma.max(x_plotlimits)
    ymin = np.ma.min(y_plotlimits)
    ymax = np.ma.max(y_plotlimits)
                        
    x_source_min = float(sCentre[0]) - JMS.ImFraction * float(lmsize)
    x_source_max = float(sCentre[0]) + JMS.ImFraction * float(lmsize)
    y_source_min = float(sCentre[1]) - JMS.ImFraction * float(lmsize)
    y_source_max = float(sCentre[1]) + JMS.ImFraction * float(lmsize)
                        
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

    # Plot ridgeline
    fig, ax = plt.subplots(figsize=(10,10))
    fig.suptitle('Source: %s' %JMS.sName)
    fig.subplots_adjust(top=0.9)
    ax.set_aspect('equal')
    ax.set_xlim(xplotmin, xplotmax)
    ax.set_ylim(yplotmin, yplotmax)
    
    A = np.ma.array(flux_array_plot, mask=np.ma.masked_invalid(flux_array_plot).mask)
    c = ax.pcolor(x, y, A, cmap=palette, vmin=JMS.vmin, vmax=JMS.vmax)

    ax.plot(ridge1[:,0], ridge1[:,1], 'r-', label='ridge 1', marker='.')
    ax.plot(ridge2[:,0], ridge2[:,1], 'r-', label='ridge 2', marker='.')
    ax.plot(sCentre[0], sCentre[1], 'g-', marker='x')   # source centre

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(c, cax = cax)
    
    fig.savefig(RLSF.Rimage %(JMS.sName, str(JMS.map_number+1)))
    plt.close(fig)
        
#############################################

def SetDataRelativeToSourcePosition(area_fluxes, ridge1, ridge2, Rlen1, Rlen2, phi_val1, phi_val2):
        
    """
    Find the source position and update data relative to this centre
    
    Parameters
    ----------
    ridge1 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for one arm of the jet

    ridge2 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for other arm of the jet

    Rlen1 - 1D array of distance from source for each ridgepoint on
             one arm of the jet

    Rlen2 - 1D array of distance from source for each ridgepoint on
            other arm of the jet

    phi_val1 - 1D array of ridgeline angles for each ridgepoint on
               one arm of the jet

    phi_val2 - 1D array of ridgeline angles for each ridgepoint on
               other arm of the jet
    
    Returns
    -------
    sCentre - 1D array, shape(2,)
              Co-ordinates of source centre

    ridge1_upd - 2D array, shape(n,2)
                 Array of ridgepoint co-ordinates for one arm of the jet

    ridge2_upd - 2D array, shape(n,2)
                 Array of ridgepoint co-ordinates for other arm of the jet

    Rlen1_upd - 1D array of distance from source for each ridgepoint on
                one arm of the jet

    Rlen2_upd - 1D array of distance from source for each ridgepoint on
                other arm of the jet

    phi_val1_upd - 1D array of ridgeline angles for each ridgepoint on
                   one arm of the jet

    phi_val2_upd - 1D array of ridgeline angles for each ridgepoint on
                   other arm of the jet
    
    """

    oldCentre = np.array([ridge1[0,0], ridge1[0,1]])                    # old centre point

    # Define points, several ridgepoints along, on either side of the jet
    point_arm1_x = ridge1[JMC.ridge_centre_search_points, 0]
    point_arm1_y = ridge1[JMC.ridge_centre_search_points, 1]
    point_arm2_x = ridge2[JMC.ridge_centre_search_points, 0]
    point_arm2_y = ridge2[JMC.ridge_centre_search_points, 1]

    # Check if we have enough ridge points in both arms. If not, retain the old centre point and don't update the data.
    if not np.isnan(point_arm1_x) and not np.isnan(point_arm2_x):

        # Find the source position
        if not isnan(JSS.sRadioRA) and not isnan(JSS.sRadioDec):

            # Source position is known in degrees
            sRA_from_mapcentre_deg = JSS.sRadioRA - JMS.sRA                 # source RA relative to map centre RA in degrees
            sDec_from_mapcentre_deg = JSS.sRadioDec - JMS.sDec              # source Dec relative to map centre Dec in degrees
            sRA_from_mapcentre_pix = sRA_from_mapcentre_deg / JMS.ddel      # source RA relative to map centre RA in pixels
            sDec_from_mapcentre_pix = sDec_from_mapcentre_deg / JMS.ddel    # source Dec relative to map centre Dec in pixels

            map_centre_pixels_x = area_fluxes.shape[1] / 2.0
            map_centre_pixels_y = area_fluxes.shape[0] / 2.0

            sCentre_x = map_centre_pixels_x + sRA_from_mapcentre_pix        # source pixel x position
            sCentre_y = map_centre_pixels_y + sDec_from_mapcentre_pix       # source pixel y position

        else:
        
            # Find the point of least flux along a line between points either side of the jet - the source pixel x/y position 
            start_point = np.array([point_arm1_x, point_arm1_y])
            end_point = np.array([point_arm2_x, point_arm2_y])
            sCentre_x, sCentre_y = JMA.GetMinimumFluxAlongLine(area_fluxes, start_point, end_point)

        sCentre = np.array([sCentre_x, sCentre_y])                          # new centre point

        # If the new centre is too close to the start or end points, a proper minimum has not been found. Retain the old 
        # centre point and don't update the data. Simiarly, if the new centre point is too close to the old centre point.
        r_newCentre_arm1 = np.sqrt( (point_arm1_x - sCentre[0])**2 + (point_arm1_y - sCentre[1])**2 )
        r_newCentre_arm2 = np.sqrt( (point_arm2_x - sCentre[0])**2 + (point_arm1_y - sCentre[1])**2 )
        rdiff_oldnew = np.sqrt((sCentre[0] - oldCentre[0])**2 + (sCentre[1] - oldCentre[1])**2)
        if r_newCentre_arm1 > 0.1 and r_newCentre_arm2 > 0.1 and rdiff_oldnew > 0.1:

            # Set all data relative to the new centre
            r_oldCentre_arm1 = np.sqrt( (point_arm1_x - oldCentre[0])**2 + (point_arm1_y - oldCentre[1])**2 )
            if r_newCentre_arm1 < r_oldCentre_arm1:                                   # Which arm does the new centre lie in?
                ridge1_upd, ridge2_upd, Rlen1_upd, Rlen2_upd, phi_val1_upd, phi_val2_upd = \
                    SetDataRelativeToCentre(sCentre, ridge1, ridge2, Rlen1, Rlen2, phi_val1, phi_val2)
            else:
                ridge2_upd, ridge1_upd, Rlen2_upd, Rlen1_upd, phi_val2_upd, phi_val1_upd = \
                    SetDataRelativeToCentre(sCentre, ridge2, ridge1, Rlen2, Rlen1, phi_val2, phi_val1)
        else:
            sCentre = oldCentre
            ridge1_upd = ridge1; ridge2_upd = ridge2
            Rlen1_upd = Rlen1; Rlen2_upd = Rlen2
            phi_val1_upd = phi_val1; phi_val2_upd = phi_val2
    else:
        sCentre = oldCentre
        ridge1_upd = ridge1; ridge2_upd = ridge2
        Rlen1_upd = Rlen1; Rlen2_upd = Rlen2
        phi_val1_upd = phi_val1; phi_val2_upd = phi_val2

    return sCentre, ridge1_upd, ridge2_upd, Rlen1_upd, Rlen2_upd, phi_val1_upd, phi_val2_upd

#############################################

def SetDataRelativeToCentre(sCentre, ridge_arm1, ridge_arm2, Rlen_arm1, Rlen_arm2, phi_val_arm1, phi_val_arm2):
        
    """
    Update data relative to source centre.
    Data for arm containing source centre is given first.
    
    Parameters
    ----------
    sCentre - 1D array, shape(2,)
              Co-ordinates of source centre

    ridge_arm1 - 2D array, shape(n,2)
                 Array of ridgepoint co-ordinates for one arm of the jet

    ridge_arm2 - 2D array, shape(n,2)
                 Array of ridgepoint co-ordinates for other arm of the jet

    Rlen_arm1 - 1D array of distance from source for each ridgepoint on
                 one arm of the jet

    Rlen_arm2 - 1D array of distance from source for each ridgepoint on
                 other arm of the jet

    phi_val_arm1 - 1D array of ridgeline angles for each ridgepoint on
                    one arm of the jet

    phi_val_arm2 - 1D array of ridgeline angles for each ridgepoint on
                    other arm of the jet
    
    Returns
    -------
    ridge_arm1_upd - 2D array, shape(n,2)
                     Array of ridgepoint co-ordinates for one arm of the jet

    ridge_arm2_upd - 2D array, shape(n,2)
                     Array of ridgepoint co-ordinates for other arm of the jet

    Rlen_arm1_upd - 1D array of distance from source for each ridgepoint on
                    one arm of the jet

    Rlen_arm2_upd - 1D array of distance from source for each ridgepoint on
                    other arm of the jet

    phi_val_arm1_upd - 1D array of ridgeline angles for each ridgepoint on
                       one arm of the jet

    phi_val_arm2_upd - 1D array of ridgeline angles for each ridgepoint on
                       other arm of the jet
    
    """
    
    # Initialise updated arrays
    ridge_arm1_upd = ridge_arm1; ridge_arm2_upd = ridge_arm2
    Rlen_arm1_upd = Rlen_arm1; Rlen_arm2_upd = Rlen_arm2
    phi_val_arm1_upd = phi_val_arm1; phi_val_arm2_upd = phi_val_arm2

    # Determine where source centre lies relative to ridge points in arm1.
    r_sCentre = np.sqrt((sCentre[0] - ridge_arm1_upd[0,0])**2 + (sCentre[1] - ridge_arm1_upd[0,1])**2)
    ridge_cnt = 0; move_count = 0; arm1_start = 0
    r_1 = 0.0
    r_2 = np.sqrt((ridge_arm1_upd[ridge_cnt+1,0] - ridge_arm1_upd[0,0])**2 + (ridge_arm1_upd[ridge_cnt+1,1] - ridge_arm1_upd[0,1])**2)
    while ridge_cnt <= JMC.ridge_centre_search_points:
        if r_1 <= r_sCentre <= r_2:
            if np.abs(r_sCentre - r_1) < 0.1:       # Source centre is too close to previous point, so replace with the centre point
                move_count = ridge_cnt - 1; arm1_start = ridge_cnt + 1 
            elif np.abs(r_sCentre - r_2) < 0.1:     # Source centre is too close to next point, so replace with the centre point
                move_count = ridge_cnt; arm1_start = ridge_cnt + 2
            else:
                move_count = ridge_cnt; arm1_start = ridge_cnt + 1 
            break

        ridge_cnt += 1
        r_1 = np.sqrt((ridge_arm1_upd[ridge_cnt,0] - ridge_arm1_upd[0,0])**2 + (ridge_arm1_upd[ridge_cnt,1] - ridge_arm1_upd[0,1])**2)
        r_2 = np.sqrt((ridge_arm1_upd[ridge_cnt+1,0] - ridge_arm1_upd[0,0])**2 + (ridge_arm1_upd[ridge_cnt+1,1] - ridge_arm1_upd[0,1])**2)

    # Update the data by moving elements from one arm array to the other.
    # Ridge points
    if move_count > 0:
        move_points = ridge_arm1_upd[1:move_count,:]
    else:
        move_points = np.empty((0,2))
    ridge_arm1_upd = np.vstack((sCentre, ridge_arm1_upd[arm1_start:,:]))
    ridge_arm2_upd = np.vstack((sCentre, move_points, ridge_arm2_upd))

    # R
    if move_count > 0:
        move_points = Rlen_arm1_upd[1:move_count]
        move_points = np.abs(move_points - r_sCentre)
    else:
        move_points = np.empty((0))
    Rlen_arm1_upd = np.hstack((np.array([r_sCentre]), Rlen_arm1_upd[arm1_start:]))
    Rlen_arm1_upd -= r_sCentre
    Rlen_arm1_upd[0] = 0.0
    Rlen_arm2_upd= np.hstack((np.array([r_sCentre]), move_points, Rlen_arm2_upd))
    Rlen_arm2_upd += r_sCentre
    Rlen_arm2_upd[0] = 0.0

    # Phi
    # Determine phi for the source centre in each arm
    point_arm1_x = ridge_arm1_upd[JMC.ridge_centre_search_points, 0]
    point_arm1_y = ridge_arm1_upd[JMC.ridge_centre_search_points, 1]
    point_arm2_x = ridge_arm2_upd[JMC.ridge_centre_search_points, 0]
    point_arm2_y = ridge_arm2_upd[JMC.ridge_centre_search_points, 1]
    phi_sCentre_arm1 = np.arctan2((point_arm1_y - sCentre[1]), (point_arm1_x - sCentre[0]))
    phi_sCentre_arm2 = np.arctan2((point_arm2_y - sCentre[1]), (point_arm2_x - sCentre[0]))
    if move_count > 0:
        move_points = phi_val_arm1_upd[1:move_count]
    else:
        move_points = np.empty((0))
    phi_val_arm1_upd = np.hstack((np.array([phi_sCentre_arm1]), phi_val_arm1_upd[arm1_start:]))
    phi_val_arm2_upd = np.hstack((np.array([phi_sCentre_arm2]), move_points, phi_val_arm2_upd))

    return ridge_arm1_upd,  ridge_arm2_upd, Rlen_arm1_upd, Rlen_arm2_upd, phi_val_arm1_upd, phi_val_arm2_upd