#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetParametersToolkit.py
Toolkit for computing parameters along a jet.
Created by LizWhitehead - 08/04/2025
"""

import JetModelling_MapSetup as JMS
import JetParameters.JetParameterFiles as JPF
import JetParameters.JPConstants as JPC
from JetParameters.JPSynchro import SynchSource
from matplotlib import pyplot as plt, ticker, axis
from astropy.cosmology import FlatLambdaCDM
import numpy as np
from numpy import pi
import os

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
    jet_parameters1 = np.empty((0,5)); jet_parameters2 = np.empty((0,5))

    # Update flux, volume and distance along the jet and to have required units
    section_parameters1 = SetRequiredUnits(section_parameters1)
    section_parameters2 = SetRequiredUnits(section_parameters2)

    ########################################################################
    # import io
    # judedata1 = []; judedata2 = []
    # with io.open("C:\Other GIT Repositories\JetContent Judith\oldtext\JudeData1.txt", mode="r", encoding="utf-8") as f:
    #     for line in f:
    #         data1 = line.split()
    #         judedata1.append([float(data1[0]), float(data1[1]), float(data1[2])])
    # with io.open("C:\Other GIT Repositories\JetContent Judith\oldtext\JudeData2.txt", mode="r", encoding="utf-8") as f:
    #     for line in f:
    #         data2 = line.split()
    #         judedata2.append([float(data2[0]), float(data2[1]), float(data2[2])])

    # jet_parameters1 = GetParametersForJetArm(judedata1)
    # jet_parameters2 = GetParametersForJetArm(judedata2)
    ########################################################################

    # Compute parameters for each arm of the jet
    jet_parameters1 = GetParametersForJetArm(section_parameters1)
    jet_parameters2 = GetParametersForJetArm(section_parameters2)

    # Save the jet parameter values to a file
    SaveParameterFiles(jet_parameters1, jet_parameters2)

    # Plot jet parameters
    PlotJetParameters(jet_parameters1, jet_parameters2)

#############################################

def GetParametersForJetArm(section_parameters):

    """
    Compute additional parameters for each section down one arm of the jet.

    Parameters
    -----------
    section_parameters - 2D array, shape(n,12)
                         Array with section points (x,y * 4), distance from source
                         and computed parameters for one arm of the jet
    
    Constants
    ---------

    Returns
    -----------
    jet_parameters - 2D array, shape(n,5)
                     Array with section points (x,y * 4), distance from source
                     and computed parameters for one arm of the jet

    Notes
    -----------
    """

    # Initialise jet parameters array
    jet_parameters = np.empty((0,5))

    # Unit conversion values
    source_R = JMS.rShift * JPC.SLight / JPC.H0                     # distance to source in kpc
    arcsec_to_kpc = source_R * np.pi/180 / 3600                     # arcsec to kpc conversion

    # Loop through section parameters array for one arm of the jet
    ##################################################################################
    for [x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, flux_section, volume_section] in section_parameters:
        R_section = (R_section_start + R_section_end) / 2           # Mid-point of section
    #for [R_section, flux_section, volume_section] in section_parameters:
    ##################################################################################

        equivSphereR = pow((volume_section / (pi * 4/3)), 1/3.0)    # Radius of equivalent sphere with volume of section
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)                       # Cosmology
        eIndex = (2 * 0.55) + 1                                     # Electron energy ("injection") index
        s = SynchSource(type='sphere', gmin=10, gmax=1e4, z=JMS.rShift, injection=eIndex, spectrum='powerlaw', cosmology=cosmo, asph=equivSphereR)
        s.normalize(freq=JMS.freq, flux=flux_section, method='equipartition', brange=(1e-11,1e-7))

        R_section_m = R_section * arcsec_to_kpc * JPC.kpc_to_m                          # section distance from source (m)
        R_section_kpc = R_section * arcsec_to_kpc                                       # section distance from source (kpc)
        volume_section_m = volume_section * pow((arcsec_to_kpc * JPC.kpc_to_m), 3)      # volume of section (m cubed)
        volume_section_kpc = volume_section * pow(arcsec_to_kpc, 3)                     # volume of section (kpc cubed)
        B_field = s.B                                                                   # magnetic field strength
        pressure = s.total_energy_density / 3.0                                         # pressure (equipartition)

        jet_parameters = np.vstack((jet_parameters, np.array([R_section_kpc, flux_section, volume_section_kpc, B_field, pressure])))

    return jet_parameters

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
        
        # Distance along the jet in arcsec
        R_section_start_= R_section_start * JMS.ddel * 3600
        R_section_end = R_section_end * JMS.ddel * 3600

        # Flux in Janskys (rather than Jy/beam)
        flux_section = flux_section / JMS.beamarea

        # Volume in arcsec cubed
        volume_section = volume_section * pow((JMS.ddel * 3600), 3)

        updated_section_parameters = np.vstack((updated_section_parameters, \
                    np.array([x1,y1, x2,y2, x3,y3, x4,y4, R_section_start, R_section_end, flux_section, volume_section])))

    return updated_section_parameters

#############################################

def SaveParameterFiles(jet_parameters1, jet_parameters2):

    """
    Saves the parameters files for each arm of the jet

    Parameters
    -----------
    jet_parameters1 - 2D array, shape(n,5)
                      Array for one arm of the jet, with distance from source 
                      and computed parameters

    jet_parameters2 - 2D array, shape(n,5)
                      Array for one arm of the jet, with distance from source 
                      and computed parameters
    
    Constants
    ---------

    Returns
    -----------

    Notes
    -----------
    """

    fileJP1 = np.column_stack((jet_parameters1[:,0], jet_parameters1[:,1], jet_parameters1[:,2], jet_parameters1[:,3], jet_parameters1[:,4]))
    fileJP2 = np.column_stack((jet_parameters2[:,0], jet_parameters2[:,1], jet_parameters2[:,2], jet_parameters2[:,3], jet_parameters2[:,4]))
    np.savetxt(JPF.JP1 %JMS.sName, fileJP1, delimiter=' ')
    np.savetxt(JPF.JP2 %JMS.sName, fileJP2, delimiter=' ')

#############################################

def PlotJetParameters(jet_parameters1, jet_parameters2):

    """
    Plots jet parameters for both arms of the jet.

    Parameters
    -----------
    jet_parameters1 - 2D array, shape(n,5)
                      Array for one arm of the jet, with distance from source 
                      and computed parameters

    jet_parameters2 - 2D array, shape(n,5)
                      Array for one arm of the jet, with distance from source 
                      and computed parameters
    
    Constants
    ---------

    Returns
    -----------

    Notes
    -----------
    """

    p_min = min(np.min(jet_parameters1[:,4]), np.min(jet_parameters2[:,4]))
    p_max = max(np.max(jet_parameters1[:,4]), np.max(jet_parameters2[:,4]))
    d_min = min(np.min(jet_parameters1[:,0]), np.min(jet_parameters2[:,0]))
    d_max = max(np.max(jet_parameters1[:,0]), np.max(jet_parameters2[:,0]))

    fig, ax = plt.subplots()
    fig.suptitle(JMS.sName)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, 105)         ## LW ## Determine automatically
    ax.set_ylim(1e-14, 1e-11)   ## LW ## Determine automatically
    axis.Axis.set_major_formatter(ax.xaxis, ticker.ScalarFormatter())

    ax.set_ylabel('Pressure (Pa)')
    ax.set_xlabel('Distance (kpc)')

    ax.plot(jet_parameters1[:,0], jet_parameters1[:,4], 'r-', label='Internal equipartition (N)', marker='.')
    ax.plot(jet_parameters2[:,0], jet_parameters2[:,4], 'g-', label='Internal equipartition (S)', marker='.')

    ax.legend()

    plt.savefig(JPF.JPimage %JMS.sName)
        
#############################################
