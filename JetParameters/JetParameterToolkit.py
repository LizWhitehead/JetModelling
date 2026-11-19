#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetParametersToolkit.py
Toolkit for computing parameters along a jet.
Created by LizWhitehead - 08/04/2025
"""

import JetModelling_MapSetup as JMS
import JetModelling_Constants as JMC
import JetModelling_SourceSetup as JSS
import JetParameters.JetParameterFiles as JPF
import JetParameters.JPConstants as JPC
from JetParameters.JPSynchro import SynchSource
from JetParameters.JPKSynch import KSynch
from matplotlib import pyplot as plt, ticker, axis
from astropy.cosmology import FlatLambdaCDM
import numpy as np
from numpy import pi
from math import nan, isnan, log10

#############################################

def GetJetParameters(section_parameters1, section_parameters2):

    """
    Compute additional parameters for each section down each side of the jet.

    Parameters
    -----------
    section_parameters1 - 2D array, shape(n,12)
                          Array with section points (x,y * 4), distance from source
                          and computed parameters for one side of the jet

    section_parameters2 - 2D array, shape(n,12)
                          Array with section points (x,y * 4), distance from source
                          and computed parameters for other side of the jet
    
    Constants
    ---------

    Returns
    -----------

    Notes
    -----------
    """

    # Initialise jet parameters arrays
    jp_kappa0_side1 = np.empty((0,8)); jp_kappa0_side2 = np.empty((0,8))

    # Update flux, volume, area and distance along the jet and to have required units
    print('Updating flux, volume and area units')
    section_parameters1 = SetRequiredUnits(section_parameters1)
    section_parameters2 = SetRequiredUnits(section_parameters2)

    # Compute parameters for each side of the jet
    print('Computing jet parameters')
    jp_kappa0_side1 = GetParametersForJetSide(section_parameters1, 1)
    jp_kappa0_side2 = GetParametersForJetSide(section_parameters2, 2)

    # Save the jet parameter values to files
    SaveParameterFiles(jp_kappa0_side1, jp_kappa0_side2)

    # Plot jet parameters
    PlotJetParameters(jp_kappa0_side1, jp_kappa0_side2)

#############################################

def GetParametersForJetSide(section_parameters, side_number):

    """
    Compute additional parameters for each section down one side of the jet.

    Parameters
    -----------
    section_parameters - 2D array, shape(n,12)
                         Array with section points (x,y * 4), distance from source
                         and computed parameters for one side of the jet

    side_number - integer, 1 or 2
                   
    
    Constants
    ---------

    Returns
    -----------
    jp_kappa0 - 2D array, shape(n,8)
                Distance from source and computed parameters for one side of the jet

    Notes
    -----------
    """

    # Initialise kappa0 array
    jp_kappa0 = np.empty((0,8))

    # Initialise external pressure
    if side_number == 1: ext_pressure_for_side = JSS.ext_pressure_side1
    else: ext_pressure_for_side = JSS.ext_pressure_side2

    # Loop through section parameters array for one side of the jet
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
            area_section_kpc = area_section * pow(arcsec_to_kpc, 2)                         # area of section (kpc squared)

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
            if not np.all(np.isnan(ext_pressure_for_side)): 
                ext_pressure = InterpolateExternalParameterValue(R_section_kpc, ext_pressure_for_side)   # external pressure

            jp_kappa0 = np.vstack((jp_kappa0, np.array([R_section_kpc, flux_section, volume_section_kpc, area_section_kpc, \
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
            print("Flux is zero in jet side " + str(side_number) + " section " + str(sect_count) + ". Update , search_angle and/or section merge parameters.")

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
                         and computed parameters for this side of the jet
    
    Constants
    ---------

    Returns
    -----------
    updated_section_parameters - 2D array, shape(n,12)
                                 Array with section points (x,y * 4), distance from source
                                 and computed parameters for this side of the jet

    Notes
    -----------
    """

    # Initialise updated section parameters array
    updated_section_parameters = np.empty((0,12))

    for [x1,y1, x2,y2, x3,y3, x4,y4, R_section, flux_section, volume_section, area_section] in section_parameters:
        
        # Distance along the jet in arcsec
        R_section = R_section * JMS.ddel * 3600

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

def SaveParameterFiles(jp_kappa0_side1, jp_kappa0_side2):

    """
    Saves the parameters files for each side of the jet

    Parameters
    -----------
    jp_kappa0_side1 - 2D array, shape(n,8)
                      Array for one side of the jet, with distance from source 
                      and computed parameters

    jp_kappa0_side2 - 2D array, shape(n,8)
                      Array for one side of the jet, with distance from source 
                      and computed parameters
    
    Constants
    ---------

    Returns
    -----------

    Notes
    -----------
    """

    np.savetxt(JPF.JP_kappa0_side1 %(JMS.sName, str(JMS.map_number+1)), jp_kappa0_side1, delimiter=' ', \
               header='section R (kpc), section flux (Jy), section volume (kpc**3), section area (kpc**2), section mag field strength (Tesla), internal pressure (Pa), emissivity, external pressure (Pa)')
    np.savetxt(JPF.JP_kappa0_side2 %(JMS.sName, str(JMS.map_number+1)), jp_kappa0_side2, delimiter=' ', \
               header='section R (kpc), section flux (Jy), section volume (kpc**3), section area (kpc**2), section mag field strength (Tesla), internal pressure (Pa), emissivity, external pressure (Pa)')

#############################################

def PlotJetParameters(jp_kappa0_side1, jp_kappa0_side2):

    """
    Plots jet parameters for both sides of the jet.

    Parameters
    -----------
    jp_kappa0_side1 - 2D array, shape(n,8)
                      Array for one side of the jet, with distance from source 
                      and computed parameters

    jp_kappa0_side2 - 2D array, shape(n,8)
                      Array for other side of the jet, with distance from source 
                      and computed parameters
    
    Constants
    ---------

    Returns
    -----------

    Notes
    -----------
    """

    lastpt1 = np.shape(jp_kappa0_side1)[0] - 1; lastpt2 = np.shape(jp_kappa0_side2)[0] - 1    # don't plot the last point

    # Model with kappa=0. Internal pressure.
    p_min = min(np.min(jp_kappa0_side1[0:lastpt1,5]), np.min(jp_kappa0_side2[0:lastpt2,5]))
    p_max = max(np.max(jp_kappa0_side1[0:lastpt1,5]), np.max(jp_kappa0_side2[0:lastpt2,5]))
    d_min = min(np.min(jp_kappa0_side1[0:lastpt1,0]), np.min(jp_kappa0_side2[0:lastpt2,0]))
    d_max = max(np.max(jp_kappa0_side1[0:lastpt1,0]), np.max(jp_kappa0_side2[0:lastpt2,0]))

    fig, ax = plt.subplots()
    fig.suptitle(JMS.sName)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(d_min, d_max)
    ax.set_ylim(p_min, p_max)
    axis.Axis.set_major_formatter(ax.xaxis, ticker.ScalarFormatter())

    ax.set_ylabel('Pressure (Pa)')
    ax.set_xlabel('Distance (kpc)')

    ax.plot(jp_kappa0_side1[0:lastpt1,0], jp_kappa0_side1[0:lastpt1,5], 'r-', label='Internal equipartition (N)', marker='.')
    ax.plot(jp_kappa0_side2[0:lastpt2,0], jp_kappa0_side2[0:lastpt2,5], 'g-', label='Internal equipartition (S)', marker='.')

    ax.legend()

    plt.savefig(JPF.JP_kappa0_plot_pressure %(JMS.sName, str(JMS.map_number+1)))
    plt.close(fig)

    # Surface brightness (flux per unit area)
    plot_data_R1 = jp_kappa0_side1[:,0]; plot_data_R2 = jp_kappa0_side2[:,0]
    plot_data_avgFlux1 = jp_kappa0_side1[:,1] / jp_kappa0_side1[:,3]; plot_data_avgFlux2 = jp_kappa0_side2[:,1] / jp_kappa0_side2[:,3]

    fa_min = min(np.min(plot_data_avgFlux1[0:lastpt1]), np.min(plot_data_avgFlux2[0:lastpt2]))
    fa_max = max(np.max(plot_data_avgFlux1[0:lastpt1]), np.max(plot_data_avgFlux2[0:lastpt2]))
    d_min = min(np.min(plot_data_R1[0:lastpt1]), np.min(plot_data_R2[0:lastpt2]))
    d_max = max(np.max(plot_data_R1[0:lastpt1]), np.max(plot_data_R2[0:lastpt2]))

    fig, ax = plt.subplots()
    fig.suptitle(JMS.sName)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(d_min, d_max)
    ax.set_ylim(fa_min, fa_max)
    axis.Axis.set_major_formatter(ax.xaxis, ticker.ScalarFormatter())

    ax.set_ylabel('Surface Brightness (Jy/kpc2)')
    ax.set_xlabel('Distance (kpc)')

    ax.plot(plot_data_R1[0:lastpt1], plot_data_avgFlux1[0:lastpt1], 'r-', label='Flux (N)', marker='.')
    ax.plot(plot_data_R2[0:lastpt2], plot_data_avgFlux2[0:lastpt2], 'g-', label='Flux (S)', marker='.')

    ax.legend()

    plt.savefig(JPF.JP_flux_area_plot %(JMS.sName, str(JMS.map_number+1)))
    plt.close(fig)
       
#############################################

def GetSpectralIndexParameters(common_section_params1, common_section_params2):

    """
    Compute the spectral index down the jet.

    Parameters
    -----------

    common_section_params1 - 3D array, shape(m,n,11)
                             Array for all maps, for one side of jet, with section points (x,y * 4),
                             distance from source, flux and area for one side of the jet.
                             n: number of sections
                             m: number of maps

    common_section_params2 - 3D array, shape(m,n,11)
                             Array for all maps, for other side of jet, with section points (x,y * 4),
                             distance from source, flux and area for other side of the jet.
                             n: number of sections
                             m: number of maps

    Returns
    -----------
    """

    print('Compute the spectral index along the jet')

    arcsec_to_kpc = JSS.angScale    # arcsec to kpc conversion (kpc/arcsec)
    
    lst_spectralindex_params1 = []          # Initialise list of spectral index parameters
    lst_spectralindex_params2 = []
    for map_number in range(JMS.map_count):
        section_params1 = np.array(common_section_params1[map_number])
        section_params2 = np.array(common_section_params2[map_number])

        # Initialise spectral index parameter arrays for map
        map_spectralindex_parameters1 = np.empty((0,3)); map_spectralindex_parameters2 = np.empty((0,3))

        for [x1,y1, x2,y2, x3,y3, x4,y4, R_section, flux_section, area_section] in section_params1:
            R_section = R_section * JMS.ddels[map_number] * 3600 * arcsec_to_kpc                    # Distance from source in kpc
            flux_section = flux_section / JMS.beamareas[map_number]                                 # Flux in Janskys (rather than Jy/beam)
            area_section = area_section * pow((JMS.ddels[map_number] * 3600 * arcsec_to_kpc), 2)    # Area in kpc squared
            map_spectralindex_parameters1 = np.vstack((map_spectralindex_parameters1, np.array([R_section, flux_section, area_section])))

        for [x1,y1, x2,y2, x3,y3, x4,y4, R_section, flux_section, area_section] in section_params2:
            R_section = R_section * JMS.ddels[map_number] * 3600 * arcsec_to_kpc                    # Distance from source in kpc
            flux_section = flux_section / JMS.beamareas[map_number]                                 # Flux in Janskys (rather than Jy/beam)
            area_section = area_section * pow((JMS.ddels[map_number] * 3600 * arcsec_to_kpc), 2)    # Area in kpc squared
            map_spectralindex_parameters2 = np.vstack((map_spectralindex_parameters2, np.array([R_section, flux_section, area_section])))

        PlotSpectralIndexSurfaceBrightness(map_number, map_spectralindex_parameters1, map_spectralindex_parameters2)
        SaveSpectralIndexFilesForMap(map_number, map_spectralindex_parameters1, map_spectralindex_parameters2)

        lst_spectralindex_params1.append(map_spectralindex_parameters1)
        lst_spectralindex_params2.append(map_spectralindex_parameters2)

    spectralindex_parameters1 = np.array(lst_spectralindex_params1)                                 # Spectral index data for all maps (side 1)
    spectralindex_parameters2 = np.array(lst_spectralindex_params2)                                 # Spectral index data for all maps (side 2)

    SaveSpectralIndexDataForSide(JPF.JP_spindex_data_side1, spectralindex_parameters1)
    SaveSpectralIndexDataForSide(JPF.JP_spindex_data_side2, spectralindex_parameters2)

    CalculateAndPlotSpectralIndexForSide(1, spectralindex_parameters1)
    CalculateAndPlotSpectralIndexForSide(2, spectralindex_parameters2)

#############################################

def PlotSpectralIndexSurfaceBrightness(map_number, spectralindex_params1, spectralindex_params2):

    """
    Plots surface brightness data for both sides of the jet.

    Parameters
    -----------
    map_number - number for the map being plotted

    spectralindex_params1 - 2D array, shape(n,3)
                            Array for one side of the jet, with distance from source, flux and area 

    spectralindex_params2 - 2D array, shape(n,3)
                            Array for other side of the jet, with distance from source, flux and area 

    Returns
    -----------
    """

    plot_data_R1 = spectralindex_params1[:,0]; plot_data_R2 = spectralindex_params2[:,0]
    plot_data_avgFlux1 = spectralindex_params1[:,1] / spectralindex_params1[:,2]
    plot_data_avgFlux2 = spectralindex_params2[:,1] / spectralindex_params2[:,2]

    fa_min = min(np.min(plot_data_avgFlux1), np.min(plot_data_avgFlux2))
    fa_max = max(np.max(plot_data_avgFlux1), np.max(plot_data_avgFlux2))
    d_min = min(np.min(plot_data_R1), np.min(plot_data_R2))
    d_max = max(np.max(plot_data_R1), np.max(plot_data_R2))

    fig, ax = plt.subplots()
    fig.suptitle(JMS.sName)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(d_min, d_max)
    ax.set_ylim(fa_min, fa_max)
    axis.Axis.set_major_formatter(ax.xaxis, ticker.ScalarFormatter())

    ax.set_ylabel('Surface Brightness (Jy/kpc2)')
    ax.set_xlabel('Distance (kpc)')

    ax.plot(plot_data_R1, plot_data_avgFlux1, 'r-', label='Flux (N)', marker='.')
    ax.plot(plot_data_R2, plot_data_avgFlux2, 'g-', label='Flux (S)', marker='.')

    ax.legend()

    plt.savefig(JPF.JP_spindex_fluxarea_plot %(JMS.sName, str(map_number+1)))
    plt.close(fig)

#############################################

def SaveSpectralIndexFilesForMap(map_number, spectralindex_params1, spectralindex_params2):

    """
    Saves the spectral index files for both sides of the jet.

    Parameters
    -----------
    map_number - map number for the data being saved

    spectralindex_params1 - 2D array, shape(n,3)
                            Array for one side of the jet, with distance from source, flux and area 

    spectralindex_params2 - 2D array, shape(n,3)
                            Array for other side of the jet, with distance from source, flux and area 

    Returns
    -----------
    """

    np.savetxt(JPF.JP_spindex_fluxarea_side1 %(JMS.sName, str(map_number+1)), spectralindex_params1, delimiter=' ',
               header='section R (kpc), section flux (Jy), section area (kpc**2)')
    np.savetxt(JPF.JP_spindex_fluxarea_side2 %(JMS.sName, str(map_number+1)), spectralindex_params2, delimiter=' ',
               header='section R (kpc), section flux (Jy), section area (kpc**2)')

#############################################

def SaveSpectralIndexDataForSide(filename, spectralindex_params):

    """
    Saves the spectral index data for one side of the jet in all maps.

    Parameters
    -----------
    filename - name of output file

    spectralindex_params - 3D array, shape(m,n,3)
                           Array for all maps, for one side of the jet, with distance from source, flux and area.
                           n: number of sections
                           m: number of maps

    Returns
    -----------
    """

    ordered_freqs = (JMS.freqs).argsort()                                                       # Order output by frequency

    # Construct output data
    sp_data = np.empty((0, (JMS.map_count * 4) + 3))                                            # Initialise the output array
    sect_count = np.size(spectralindex_params, 1)                                               # Section count
    for sect_num in range(sect_count):
        line = np.array([spectralindex_params[0,sect_num,0]])                                   # Distance from source (kpc)
        spindex_data = np.empty((0,2))                                                          # Data for calculation of section spectral index
        for map_number in ordered_freqs:
            norm_area = spectralindex_params[map_number,0,2]                                    # Normalising area
            norm_factor = norm_area / spectralindex_params[map_number,sect_num,2]               # Normalisation factor

            freq = JMS.freqs[map_number] / 1e6                                                  # Freq (MHz)
            log_freq = log10(freq)                                                              # Log freq
            section_flux = spectralindex_params[map_number,sect_num,1] * norm_factor            # Flux (Jy)
            log_section_flux = log10(section_flux)                                              # Log flux

            flux_for_freq = np.array([freq, log_freq, section_flux, log_section_flux])          # All freq and flux data for this map section
            logfreq_logflux = np.array([log_freq,log_section_flux])                             # Freq and flux log values for this map section
            
            line = np.hstack((line, flux_for_freq))
            spindex_data = np.vstack((spindex_data, logfreq_logflux))

        logfreq = spindex_data[:,0]; logflux = spindex_data[:,1]
        grad_intercept, cov = np.polyfit(logfreq, logflux, 1, cov=True)
        gradient = grad_intercept[0]; std_dev = np.sqrt(cov[0][0])
        spindex = - gradient                                                                    # Spectral index for single section

        line = np.hstack((line, spindex, std_dev))
        sp_data = np.vstack((sp_data, line))

    # Header
    header_text = "section R (kpc)"
    for map_number in range(JMS.map_count): header_text += ", freq(MHz), log freq, flux (Jy), log flux"
    header_text += ", spectral index, std deviation"

    np.savetxt(filename %(JMS.sName), sp_data, delimiter=' ', header=header_text)

#############################################

def CalculateAndPlotSpectralIndexForSide(side_number, spectralindex_params):

    """
    Divide the jet into lengths, for which to find the spectral index. 
    Plot log flux vs log freq and determine the spectral index for each of these lengths.

    Parameters
    -----------
    side_number - number of the jet side being processed

    spectralindex_params - 3D array, shape(m,n,3)
                           Array for all maps, for one side of the jet, with distance from source, flux and area.
                           n: number of sections
                           m: number of maps

    Returns
    -----------
    """

    SPINDEX_INNER_JET_LENGTH = 10                               # Length of the inner jet to ignore in spectral index calculation (kpc)

    # Initialise output file names
    if side_number == 1:
        calculated_spindex_filename = JPF.JP_spectralindex_side1
        spindex_data_plot_filename = JPF.JP_spectralindex_plot_side1
    else:
        calculated_spindex_filename = JPF.JP_spectralindex_side2
        spindex_data_plot_filename = JPF.JP_spectralindex_plot_side2

    section_Rs = np.array(spectralindex_params[0,:,0])          # Distance from source for all jet sections (kpc)

    # Find the first section which has R greater than the inner jet length
    for sect_num in range(np.size(section_Rs)):
        if section_Rs[sect_num] >= SPINDEX_INNER_JET_LENGTH:
            sect_first = sect_num
            R_start = section_Rs[sect_num]
            sect_start = sect_first
            break

    # Divide the jet into lengths, to find the spectral index
    length_spindex_data = np.empty((0,2))                                                       # Initialise spectral index data for jet length
    calc_spindex_data = np.empty((0,4))                                                         # Initialise calculated spectral index data
    for sect_num in range(sect_first, np.size(section_Rs)):
        sect_spindex_data = np.empty((0,2))                                                     # Initialise spectral index data for jet section

        for map_number in range(JMS.map_count):
            norm_area = spectralindex_params[map_number,0,2]                                    # Normalising area
            norm_factor = norm_area / spectralindex_params[map_number,sect_num,2]               # Normalisation factor

            freq = JMS.freqs[map_number] / 1e6                                                  # Freq (MHz)
            log_freq = log10(freq)                                                              # Log freq
            section_flux = spectralindex_params[map_number,sect_num,1] * norm_factor            # Flux (Jy)
            log_section_flux = log10(section_flux)                                              # Log flux

            logfreq_logflux = np.array([log_freq,log_section_flux])                             # Freq and flux log values for this map section
            sect_spindex_data = np.vstack((sect_spindex_data, logfreq_logflux))

        if (sect_num - sect_start) < JMC.spindex_sections_in_length:
            length_spindex_data = np.vstack((length_spindex_data, sect_spindex_data))

            if sect_num == np.size(section_Rs) - 1:                                             # Check whether we've reached the end of the jet
                R_end = section_Rs[sect_num]
                logfreq = length_spindex_data[:,0]; logflux = length_spindex_data[:,1]
                grad_intercept, cov = np.polyfit(logfreq, logflux, 1, cov=True)
                gradient = grad_intercept[0]; intercept = grad_intercept[1]; std_dev = np.sqrt(cov[0][0])
                spindex = - gradient                                                            # Spectral index for this length of the jet

                PlotSpectralIndexData(spindex_data_plot_filename, spindex, std_dev, logfreq, logflux, gradient, intercept, R_start, R_end)
                calc_spindex_data = np.vstack((calc_spindex_data, np.array([R_start, R_end, spindex, std_dev])))
        else:
            R_end = section_Rs[sect_num]
            logfreq = length_spindex_data[:,0]; logflux = length_spindex_data[:,1]
            grad_intercept, cov = np.polyfit(logfreq, logflux, 1, cov=True)
            gradient = grad_intercept[0]; intercept = grad_intercept[1]; std_dev = np.sqrt(cov[0][0])
            spindex = - gradient                                                                # Spectral index for this length of the jet

            PlotSpectralIndexData(spindex_data_plot_filename, spindex, std_dev, logfreq, logflux, gradient, intercept, R_start, R_end)
            calc_spindex_data = np.vstack((calc_spindex_data, np.array([R_start, R_end, spindex, std_dev])))

            length_spindex_data = sect_spindex_data                                             # Re-initialise spectral index data for jet length
            R_start = section_Rs[sect_num]
            sect_start = sect_num

    # Save the calculated spectral indices for the jet lengths
    np.savetxt(calculated_spindex_filename %(JMS.sName), calc_spindex_data, delimiter=' ', header='R_start(kpc), R_end(kpc), spectral index, std deviation')

#############################################

def PlotSpectralIndexData(plot_filename, spectral_index, std_dev, logfreq, logflux, gradient, intercept, R_start, R_end):

    """
    Plot the spectral index data - log flux density vs log frequency.

    Parameters
    -----------
    plot_filename - name of the output plot file

    spectral_index - spectral index value

    std_dev - spectral index value standard deviation

    logfreq - log frequency (x) values

    logflux - log flux density (y) values

    gradient, intercept - gradient, intercept of best fit straight line

    R_start, R_end - start and end distances from source for this jet length

    Returns
    -----------
    """

    plt.figure()

    # Plot the data points
    plt.plot(logfreq, logflux, 'o', label='Spectral index = ' + str(spectral_index) + ' +/- ' + format(std_dev, ".2f"), markersize=4, c='black')

    # Plot the best fit straight line
    min_x = np.floor(np.min(logfreq) * 2) / 2; max_x = np.ceil(np.max(logfreq) * 2) / 2
    x = np.linspace(min_x, max_x, 10)
    plt.plot(x, (gradient * x) + intercept, c='red')

    plt.xlabel('log frequency')
    plt.ylabel('log flux density')

    plt.legend()

    plt.savefig(plot_filename %(JMS.sName, format(R_start, ".1f"), format(R_end, ".1f")))
