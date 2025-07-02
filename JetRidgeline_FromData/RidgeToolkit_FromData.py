# RidgeToolkit_FromData.py
# Toolkit for loading existing ridgeline data
# into internal data structures, required for
# edgepoint and section creation.
# Created by LizWhitehead - 21/06/2025

import JetModelling_MapSetup as JMS
import JetModelling_Constants as JMC
import JetModelling_MapAnalysis as JMA
import JetRidgeline_FromData.RidgelineFiles_FromData as RLFDF
import io
import numpy as np
from math import atan2, isnan
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

#############################################

def LoadRidgelineData(flux_array):

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
    ridge1, phi_val1, Rlen1 = LoadRidgelineDataForOneArm(JMC.ridgelines_from_data_arm1, ridge1, phi_val1, Rlen1)

    # Read ridgeline data file for other arm
    ridge2 = np.empty((0,2)); phi_val2 = np.empty((0)); Rlen2 = np.empty((0))
    ridge2, phi_val2, Rlen2 = LoadRidgelineDataForOneArm(JMC.ridgelines_from_data_arm2, ridge2, phi_val2, Rlen2)

    # Determine the source position and update all data to be relative to this position
    sCentre, ridge1, ridge2, Rlen1, Rlen2, phi_val1, phi_val2 = \
        SetDataRelativeToSourcePosition(flux_array, ridge1, ridge2, Rlen1, Rlen2, phi_val1, phi_val2)

    # Save files and plot data
    SaveRidgelineFiles(ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2)
    PlotRidgelines(flux_array, sCentre, ridge1, ridge2)

    return ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2

#############################################

def LoadRidgelineDataForOneArm(input_file_name, ridge, phi_val, Rlen):

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
    with io.open(input_file_name, mode="r", encoding="utf-8") as file1:
        for line in file1:
            data = line.split()

            if lineidx == 0:
                x = float(data[0]); y = float(data[1])
                ridge = np.vstack((ridge, np.array([x, y])))
                Rlen = np.hstack((Rlen, 0.0))
                last_saved_x = x; last_saved_y = y; last_saved_R = 0.0
            else:
                x = float(data[0]); y = float(data[1])
                latest_R += np.sqrt( (x - last_saved_x)**2 + (y - last_saved_y)**2 )
                if (latest_R - last_saved_R) > JMC.R_fd:                 # Only save when difference in R > defined max value
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
    np.savetxt(RLFDF.R1 %JMS.sName, file1, delimiter=' ', \
               header='ridgepoint x-coord (pix), ridgepoint y-coord (pix), ridgepoint phi (radns), ridgepoint R (pix)')
    np.savetxt(RLFDF.R2 %JMS.sName, file2, delimiter=' ', \
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

    y_plotlimits = np.ma.masked_array(y, mask=np.ma.masked_where(y < (JMS.nSig * JMS.bgRMS), y, copy=True).mask)
    x_plotlimits = np.ma.masked_array(x, np.ma.masked_where(x < (JMS.nSig * JMS.bgRMS), x, copy=True).mask)
    xmin = np.ma.min(x_plotlimits)
    xmax = np.ma.max(x_plotlimits)
    ymin = np.ma.min(y_plotlimits)
    ymax = np.ma.max(y_plotlimits)
                        
    x_source_min = float(sCentre[0]) - JMC.ImFraction * float(lmsize)
    x_source_max = float(sCentre[0]) + JMC.ImFraction * float(lmsize)
    y_source_min = float(sCentre[1]) - JMC.ImFraction * float(lmsize)
    y_source_max = float(sCentre[1]) + JMC.ImFraction * float(lmsize)
                        
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
    c = ax.pcolor(x, y, A, cmap=palette, vmin=JMC.vmin, vmax=JMC.vmax)

    ax.plot(ridge1[:,0], ridge1[:,1], 'r-', label='ridge 1', marker='.')
    ax.plot(ridge2[:,0], ridge2[:,1], 'r-', label='ridge 2', marker='.')
    ax.plot(sCentre[0], sCentre[1], 'g-', marker='x')   # source centre

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(c, cax = cax)
    
    fig.savefig(RLFDF.Rimage %JMS.sName)
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
        if not isnan(JMS.sRadioRA) and not isnan(JMS.sRadioDec):

            # Source position is known in degrees
            sRA_from_mapcentre_deg = JMS.sRadioRA - JMS.sRA                 # source RA relative to map centre RA in degrees
            sDec_from_mapcentre_deg = JMS.sRadioDec - JMS.sDec              # source Dec relative to map centre Dec in degrees
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