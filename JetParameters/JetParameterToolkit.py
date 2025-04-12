#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetParametersToolkit.py
Toolkit for calculating parameters along a jet.
Created by LizWhitehead - 08/04/2025
"""

import JetParameters.JetParameterFiles as JPF
import JetParameters.JPConstants as JPC
import JetParameters.JPGlobal as JPG
import numpy as np
from numpy import pi

#############################################

def GetJetParameters(section_parameters1, section_parameters2):

    """
    Compute additional parameters for each section down each arm of the jet.

    Parameters
    -----------
    section_parameters1 - 2D array, shape(n,12)
                          Array with section points (x,y * 4), distance from source
                          and computed parameters for one arm of the jet

    section_parameters2 - 2D array, shape(n,12)
                          Array with section points (x,y * 4), distance from source
                          and computed parameters for other arm of the jet
    
    Constants
    ---------

    Returns
    -----------

    Notes
    -----------
    """

    # Initialise jet parameters arrays
    jet_parameters1 = np.empty((0,3)); jet_parameters2 = np.empty((0,3))

    # Update flux, volume and distance along the jet and to have required units
    # section_parameters1 = SetRequiredUnits(section_parameters1)
    # section_parameters2 = SetRequiredUnits(section_parameters2)

    # Loop through section parameters array for one arm of the jet
    for [x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, flux_section, volume_section] in section_parameters1:
        R_section = (R_section_start + R_section_end) / 2       # Mid-point of section
        jet_parameters1 = np.vstack((jet_parameters1, np.array([R_section, flux_section, volume_section])))

    # Loop through section parameters array for other arm of the jet
    for [x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, flux_section, volume_section] in section_parameters2:
        R_section = (R_section_start + R_section_end) / 2       # Mid-point of section
        jet_parameters2 = np.vstack((jet_parameters2, np.array([R_section, flux_section, volume_section])))

    # Save the jet parameter values to a file
    SaveParameterFiles(jet_parameters1, jet_parameters2)

#############################################

def SetRequiredUnits(section_parameters):

    """
    Returns the merged section array with distance, flux and volume
    in the correct units.

    Parameters
    -----------
    section_parameters - 2D array, shape(n,12)
                         Array with section points (x,y * 4), distance from source
                         and computed parameters for other arm of the jet
    
    Constants
    ---------

    Returns
    -----------
    updated_section_parameters - 2D array, shape(n,12)
                                 Array with section points (x,y * 4), distance from source
                                 and computed parameters for other arm of the jet

    Notes
    -----------
    """

    # Initialise updated section parameters array
    updated_section_parameters = np.empty((0,12))

    for [x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, flux_section, volume_section] in section_parameters:
        
        # Distance to the source in kpc
        source_r = JPG.rShift * JPC.SLight / JPC.H0

        # Distance along the jet in kpc
        R_section_start_rdns = R_section_start * JPC.ddel * pi/180
        R_section_start  = R_section_start_rdns * source_r
        R_section_end_rdns = R_section_end * JPC.ddel * pi/180
        R_section_end  = R_section_end_rdns * source_r

        # Flux in Janskys (rather than Jy/beam)
        flux_section = flux_section / JPC.beamarea

        # Volume in kpc cubed
        volume_section = volume_section * pow((JPC.ddel * pi/180 * source_r), 3)

        updated_section_parameters = np.vstack((updated_section_parameters, \
                    np.array([x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, flux_section, volume_section])))

    return updated_section_parameters

#############################################

def SaveParameterFiles(jet_parameters1, jet_parameters2):

    """
    Saves the parameters files for each arm of the jet

    Parameters
    -----------
    source_name - str,
                  the name of the source

    jet_parameters1 - 2D array, shape(n,12)
                      Array for one arm of the jet, with distance from source 
                      and computed parameters

    jet_parameters2 - 2D array, shape(n,12)
                      Array for one arm of the jet, with distance from source 
                      and computed parameters
    
    Constants
    ---------

    Returns
    -----------

    Notes
    -----------
    """

    try:
        fileJP1 = np.column_stack((jet_parameters1[:,0], jet_parameters1[:,1], jet_parameters1[:,2]))
        fileJP2 = np.column_stack((jet_parameters2[:,0], jet_parameters2[:,1], jet_parameters2[:,2]))
        np.savetxt(JPF.JP1 %JPG.sName, fileJP1, delimiter=' ')
        np.savetxt(JPF.JP2 %JPG.sName, fileJP2, delimiter=' ')
    except Exception as e:
        print('Error occurred saving jet parameters files')

#############################################

