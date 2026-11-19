#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RidgeToolkit_FromData.py
Toolkit for loading existing ridgeline data
into internal data structures, required for
edgepoint and section creation.
Created by LizWhitehead - 21/06/2025
"""

import JetModelling_MapSetup as JMS
import JetModelling_Constants as JMC
import JetModelling_MapAnalysis as JMA
import JetRidgeline_FromData.RidgelineFiles_FromData as RLFDF
import io
import numpy as np
from math import atan2, sin, cos, floor
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
             Array of ridgepoint co-ordinates for one side of the jet
    
    ridge2 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for other side of the jet

    phi_val1 - 1D array of ridgeline angles for each ridgepoint on
               one side of the jet
    
    phi_val2 - 1D array of ridgeline angles for each ridgepoint on
               other side of the jet

    Rlen1 - 1D array of distance from source for each ridgepoint on
            one side of the jet
    
    Rlen2 - 1D array of distance from source for each ridgepoint on
            other side of the jet
    """
    # Read ridgeline data file for first side
    ridge1 = np.empty((0,2)); phi_val1 = np.empty((0)); Rlen1 = np.empty((0))
    ridge1, phi_val1, Rlen1 = LoadRidgelineDataForOneSide(JMS.ridgelines_from_data_side1, ridge1, phi_val1, Rlen1)

    # Read ridgeline data file for other side
    ridge2 = np.empty((0,2)); phi_val2 = np.empty((0)); Rlen2 = np.empty((0))
    ridge2, phi_val2, Rlen2 = LoadRidgelineDataForOneSide(JMS.ridgelines_from_data_side2, ridge2, phi_val2, Rlen2)

    # Save files and plot data
    SaveRidgelineFiles(ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2)
    PlotRidgelines(flux_array, ridge1, ridge2)

    return ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2

#############################################

def LoadRidgelineDataForOneSide(input_file_name, ridge, phi_val, Rlen):

    """
    Load existing ridgeline data into internal data
    structures, for one side of the jet

    Parameters
    -----------
    input_file_name - string
                      Name of input text file

    ridge - 2D array, shape(n,2)
            Array of ridgepoint co-ordinates for one side of the jet

    phi_val - 1D array of ridgeline angles for each ridgepoint on
              one side of the jet

    Rlen - 1D array of distance from source for each ridgepoint on
           one side of the jet

    Returns
    -----------
    ridge - 2D array, shape(n,2)
            Array of ridgepoint co-ordinates for one side of the jet

    phi_val - 1D array of ridgeline angles for each ridgepoint on
              one side of the jet

    Rlen - 1D array of distance from source for each ridgepoint on
           one side of the jet
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
                latest_phi = atan2((y - last_saved_y), (x - last_saved_x))

                while (latest_R - last_saved_R) > JMS.R:                     # Only save when difference in R > defined max value
                    fraction_R = float(JMS.R / (latest_R - last_saved_R))
                    delta_R = ((latest_R - last_saved_R) * fraction_R)
                    delta_x = delta_R * cos(latest_phi)
                    delta_y = delta_R * sin(latest_phi)

                    x_save = last_saved_x + delta_x
                    y_save = last_saved_y + delta_y
                    R_save = last_saved_R + delta_R

                    ridge = np.vstack((ridge, np.array([x_save, y_save])))
                    Rlen = np.hstack((Rlen, R_save))
                    phi_val = np.hstack(( phi_val, latest_phi ))
                    last_saved_x = x_save; last_saved_y = y_save; last_saved_R = R_save

            lineidx += 1

        # Save the last data if necessary
        if x != last_saved_x or y != last_saved_y:
            ridge = np.vstack((ridge, np.array([x, y])))
            Rlen = np.hstack((Rlen, latest_R))
            phi_val = np.hstack(( phi_val, latest_phi ))

        # Add a final phi value, the same as the previous one
        phi_val = np.hstack((phi_val, phi_val[-1]))

        # Add a null row
        ridge = np.vstack((ridge, np.array([np.nan, np.nan])))
        Rlen = np.hstack((Rlen, np.nan))
        phi_val = np.hstack((phi_val, np.nan))

        # Update co-ordinates to be relative to the cutout
        cutout_offset = JMA.GetCutoutOffset(JMS.mCentre, np.array([JMS.sSize[1],JMS.sSize[0]]))
        ridge -= cutout_offset 

    return ridge, phi_val, Rlen

#############################################

def SaveRidgelineFiles(ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2):

    """
    Saves the ridgeline files for each side of the jet

    Parameters
    -----------
    ridge1 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for one side of the jet
    
    ridge2 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for other side of the jet

    phi_val1 - 1D array of ridgeline angles for each ridgepoint on
               one side of the jet
    
    phi_val2 - 1D array of ridgeline angles for each ridgepoint on
               other side of the jet

    Rlen1 - 1D array of distance from source for each ridgepoint on
            one side of the jet
    
    Rlen2 - 1D array of distance from source for each ridgepoint on
            other side of the jet
    """

    file1 = np.column_stack((ridge1[:,0], ridge1[:,1], phi_val1, Rlen1))
    file2 = np.column_stack((ridge2[:,0], ridge2[:,1], phi_val2, Rlen2))
    np.savetxt(RLFDF.R1 %(JMS.sName, str(JMS.map_number+1)), file1, delimiter=' ', \
               header='ridgepoint x-coord (pix), ridgepoint y-coord (pix), ridgepoint phi (radns), ridgepoint R (pix)')
    np.savetxt(RLFDF.R2 %(JMS.sName, str(JMS.map_number+1)), file2, delimiter=' ', \
               header='ridgepoint x-coord (pix), ridgepoint y-coord (pix), ridgepoint phi (radns), ridgepoint R (pix)')

#############################################

def PlotRidgelines(flux_array, ridge1, ridge2):

    """
    Plots the edge points and jet sections on the source.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    ridge1 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for one side of the jet
    
    ridge2 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for other side of the jet

    """

    # Amplify flux by specfied factor
    flux_array_plot = JMC.flux_factor * flux_array.copy()

    palette = plt.cm.cividis
    palette = copy.copy(plt.get_cmap("cividis"))
    palette.with_extremes(bad='k')

    y, x = np.mgrid[slice((0),(flux_array_plot.shape[0]),1), slice((0),(flux_array_plot.shape[1]),1)]
    y = np.ma.masked_array(y, mask=np.ma.masked_invalid(flux_array_plot).mask)
    x = np.ma.masked_array(x, mask=np.ma.masked_invalid(flux_array_plot).mask)

    # Reduce plotted region by specified factor
    x_pixels_not_plotted = (1.0 - JMS.ImFraction) * float(JMS.sSize[0])
    xplotmin = x_pixels_not_plotted / 2.0; xplotmax = float(JMS.sSize[0]) - (x_pixels_not_plotted / 2.0)
    y_pixels_not_plotted = (1.0 - JMS.ImFraction) * float(JMS.sSize[1])
    yplotmin = y_pixels_not_plotted / 2.0; yplotmax = float(JMS.sSize[1]) - (y_pixels_not_plotted / 2.0)

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
    ax.plot(ridge1[0,0], ridge1[0,1], 'g-', marker='x', ms=10)   # source centre

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(c, cax = cax)
    
    fig.savefig(RLFDF.Rimage %(JMS.sName, str(JMS.map_number+1)))
    plt.close(fig)