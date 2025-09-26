#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetParametersToolkit.py
Toolkit for computing parameters along a jet.
Created by LizWhitehead - 08/04/2025
"""

import JetModelling_MapSetup as JMS
import JetModelling_SourceSetup as JSS
import JetParameters.JetParameterFiles as JPF
import JetParameters.JPConstants as JPC
from JetParameters.JPSynchro import SynchSource
from JetParameters.JPKSynch import KSynch
from matplotlib import pyplot as plt, ticker, axis
from astropy.cosmology import FlatLambdaCDM
import numpy as np
from numpy import pi
from math import nan, isnan

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
    jp_kappa0_arm1 = np.empty((0,7)); jp_kappa0_arm2 = np.empty((0,7))

    # Update flux, volume, area and distance along the jet and to have required units
    print('Updating flux, volume and area units')
    section_parameters1 = SetRequiredUnits(section_parameters1)
    section_parameters2 = SetRequiredUnits(section_parameters2)

    # Compute parameters for each arm of the jet
    print('Computing jet parameters')
    jp_kappa0_arm1 = GetParametersForJetArm(section_parameters1, 1)
    jp_kappa0_arm2 = GetParametersForJetArm(section_parameters2, 2)

    # Save the jet parameter values to files
    SaveParameterFiles(jp_kappa0_arm1, jp_kappa0_arm2)

    # Plot jet parameters
    PlotJetParameters(jp_kappa0_arm1, jp_kappa0_arm2)

#############################################

def GetParametersForJetArm(section_parameters, arm_number):

    """
    Compute additional parameters for each section down one arm of the jet.

    Parameters
    -----------
    section_parameters - 2D array, shape(n,12)
                         Array with section points (x,y * 4), distance from source
                         and computed parameters for one arm of the jet

    arm_number - integer, 1 or 2
                   
    
    Constants
    ---------

    Returns
    -----------
    jp_kappa0 - 2D array, shape(n,7)
                Distance from source and computed parameters for one arm of the jet

    Notes
    -----------
    """

    # Initialise kappa0 array
    jp_kappa0 = np.empty((0,7))

    # Initialise external pressure
    if arm_number == 1: ext_pressure_for_arm = JSS.ext_pressure_arm1
    else: ext_pressure_for_arm = JSS.ext_pressure_arm2

    # Loop through section parameters array for one arm of the jet
    sect_count = 0
    for [x1,y1, x2,y2, x3,y3, x4,y4, R_section, flux_section, volume_section, area_section] in section_parameters:
        sect_count += 1

        # Check that the total flux in this section is greater that the background RMS value
        if flux_section > (JMS.bgRMS / JMS.beamarea):

            # source_R = JSS.rShift * JPC.SLight / JPC.H0
            # angular_scale = source_R * pi/180 / 3600
            arcsec_to_kpc = JSS.angScale                                                    # arcsec to kpc conversion (kpc/arcsec)
            cosmo = FlatLambdaCDM(H0=70, Om0=0.3)                                           # Cosmology
            R_section_m = R_section * arcsec_to_kpc * JPC.kpc_to_m                          # section distance from source (m)
            R_section_kpc = R_section * arcsec_to_kpc                                       # section distance from source (kpc)
            volume_section_m = volume_section * pow((arcsec_to_kpc * JPC.kpc_to_m), 3)      # volume of section (m cubed)
            volume_section_kpc = volume_section * pow(arcsec_to_kpc, 3)                     # volume of section (kpc cubed)

            ################################################################################################################
            # Model with kappa=0, at equipartition. Find internal pressure, magnetic field and emissivity.
            ################################################################################################################

            eIndex = (2 * JSS.spectral_index) + 1                       # Electron energy ("injection") index
            equivSphereR = pow((volume_section / (pi * 4/3)), 1/3.0)    # Radius of equivalent sphere with volume of section
            kappa = 0.0                                                 # Assume no non-radiating particles
            gamma_min = 10                                              # Electron spectrum power law minimum gamma
            gamma_max = 1e4                                             # Electron spectrum power law maximum gamma
            min_field = 1e-11                                           # Minimum field strength
            max_field = 1e-7                                            # Maximum field strength

            s = SynchSource(type='sphere', gmin=gamma_min, gmax=gamma_max, z=JSS.rShift, injection=eIndex, 
                            spectrum='powerlaw', cosmology=cosmo, asph=equivSphereR)
            s.normalize(freq=JMS.freq, flux=flux_section, zeta=(1 + kappa), method='equipartition', brange=(min_field,max_field))

            B_field = s.B                                               # magnetic field strength
            int_pressure = s.total_energy_density / 3.0                 # internal pressure (equipartition)
            emissivity = s.emiss(JMS.freq)                              # emissivity

            ext_pressure = nan
            if not np.all(np.isnan(ext_pressure_for_arm)): 
                ext_pressure = InterpolateExternalParameterValue(R_section_kpc, ext_pressure_for_arm)   # external pressure

            jp_kappa0 = np.vstack((jp_kappa0, np.array([R_section_kpc, flux_section, volume_section_kpc, \
                                                        B_field, int_pressure, emissivity, ext_pressure])))
        

            ################################################################################################################
            # Model to combine internal and external pressure and estimate a profile of kappa along the jet.
            ################################################################################################################

            if not isnan(ext_pressure):
                k = KSynch()
                k.ksynch_calculate(JMS.freq, emissivity, ext_pressure)
                beqnoprot = k.beqnoprot; pintnoprot = k.pintnoprot; krel = k.krel; beqrprot = k.beqrprot; uer = k.uer
                ubr = k.ubr; ur = k.ur; kval = k.kval; beqtprot = k.beqtprot; uet = k.uet; ubt = k.ubt; uth = k.uth
                bdme = k.bdome; uedome = k.uedome; ubdome = k.ubdome; bdomb = k.bdomb; uedomb = k.uedomb; ubdomb = k.ubdomb

        else:
            print("Flux is zero in jet arm " + str(arm_number) + " section " + str(sect_count) + ". Update ridge_centre_search_points, nSig_arms and/or section merge parameters.")

    return jp_kappa0

#############################################

def InterpolateExternalParameterValue(section_r, ext_parameter):

    """
    Interpolate the external parameter value, corresponding to the input distance
    along the jet.

    Parameters
    -----------
    section_r - float
                Distance along the jet

    ext_parameter - 2D array, shape(n,2)
                    Array of distance along the jet and parameter value
                   
    
    Constants
    ---------

    Returns
    -----------
    ext_param_value - interpolated parameter value (-1 if input distance not within range)

    Notes
    -----------
    """

    ext_param_value = nan

    icnt = 0
    while icnt < np.shape(ext_parameter)[0]:
        if icnt == 0:
            if section_r < ext_parameter[icnt,0]:
                break
        else:
            if section_r < ext_parameter[icnt,0]:
                rdiff_fraction = (section_r - ext_parameter[icnt-1,0]) / (ext_parameter[icnt,0] - ext_parameter[icnt-1,0])
                ext_param_value = ext_parameter[icnt-1,1] + (rdiff_fraction * (ext_parameter[icnt,1] - ext_parameter[icnt-1,1]))
                break
        icnt += 1

    return ext_param_value

#############################################

def SetRequiredUnits(section_parameters):

    """
    Returns the merged section array with distance, flux and volume
    in the correct units.

    Parameters
    -----------
    section_parameters - 2D array, shape(n,12)
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

    for [x1,y1, x2,y2, x3,y3, x4,y4, R_section, flux_section, volume_section, area_section] in section_parameters:
        
        # Distance along the jet in arcsec
        R_section_= R_section * JMS.ddel * 3600

        # Flux in Janskys (rather than Jy/beam)
        flux_section = flux_section / JMS.beamarea

        # Volume in arcsec cubed
        volume_section = volume_section * pow((JMS.ddel * 3600), 3)

        # Area in arcsec squared
        area_section = area_section * pow((JMS.ddel * 3600), 2)

        updated_section_parameters = np.vstack((updated_section_parameters, \
                    np.array([x1,y1, x2,y2, x3,y3, x4,y4, R_section, flux_section, volume_section, area_section])))

    return updated_section_parameters

#############################################

def SaveParameterFiles(jp_kappa0_arm1, jp_kappa0_arm2):

    """
    Saves the parameters files for each arm of the jet

    Parameters
    -----------
    jp_kappa0_arm1 - 2D array, shape(n,7)
                     Array for one arm of the jet, with distance from source 
                     and computed parameters

    jp_kappa0_arm2 - 2D array, shape(n,7)
                     Array for one arm of the jet, with distance from source 
                     and computed parameters
    
    Constants
    ---------

    Returns
    -----------

    Notes
    -----------
    """

    np.savetxt(JPF.JP_kappa0_arm1 %(JMS.sName, str(JMS.map_number+1)), jp_kappa0_arm1, delimiter=' ', \
               header='section R (kpc), section flux (Jy), section volume (kpc**3), section mag field strength (Tesla), internal pressure (Pa), emissivity, external pressure (Pa)')
    np.savetxt(JPF.JP_kappa0_arm2 %(JMS.sName, str(JMS.map_number+1)), jp_kappa0_arm2, delimiter=' ', \
               header='section R (kpc), section flux (Jy), section volume (kpc**3), section mag field strength (Tesla), internal pressure (Pa), emissivity, external pressure (Pa)')

#############################################

def PlotJetParameters(jp_kappa0_arm1, jp_kappa0_arm2):

    """
    Plots jet parameters for both arms of the jet.

    Parameters
    -----------
    jp_kappa0_arm1 - 2D array, shape(n,7)
                     Array for one arm of the jet, with distance from source 
                     and computed parameters

    jp_kappa0_arm2 - 2D array, shape(n,7)
                     Array for one arm of the jet, with distance from source 
                     and computed parameters
    
    Constants
    ---------

    Returns
    -----------

    Notes
    -----------
    """

    # Model with kappa=0. Internal pressure.
    p_min = min(np.min(jp_kappa0_arm1[:,4]), np.min(jp_kappa0_arm2[:,4]))
    p_max = max(np.max(jp_kappa0_arm1[:,4]), np.max(jp_kappa0_arm2[:,4]))
    d_min = min(np.min(jp_kappa0_arm1[:,0]), np.min(jp_kappa0_arm2[:,0]))
    d_max = max(np.max(jp_kappa0_arm1[:,0]), np.max(jp_kappa0_arm2[:,0]))

    fig, ax = plt.subplots()
    fig.suptitle(JMS.sName)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(d_min, d_max)
    ax.set_ylim(p_min, p_max)
    axis.Axis.set_major_formatter(ax.xaxis, ticker.ScalarFormatter())

    ax.set_ylabel('Pressure (Pa)')
    ax.set_xlabel('Distance (kpc)')

    ax.plot(jp_kappa0_arm1[:,0], jp_kappa0_arm1[:,4], 'r-', label='Internal equipartition (N)', marker='.')
    ax.plot(jp_kappa0_arm2[:,0], jp_kappa0_arm2[:,4], 'g-', label='Internal equipartition (S)', marker='.')

    ax.legend()

    plt.savefig(JPF.JP_kappa0_plot_pressure %(JMS.sName, str(JMS.map_number+1)))
        
#############################################
