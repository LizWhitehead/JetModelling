#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetSections.py
Divide both sides of the jet into sections.
Created by LizWhitehead - 30/04/2025
"""

import JetSections.JSSetup as JSS
from JetSections.JetSectionToolkit import GetEdgepointsAndSections, CommonJetSectionsAcrossMultipleMaps
import time

def CreateJetSections(flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2):

    """
    Divide both sides of the jet into sections by finding edge points.

    Parameters
    -----------
    flux_array - 2D array,
                 raw image array

    ridge1 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for one side of the jet
    
    ridge2 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for other side of the jet

    phi_val1 - 1D array of ridgeline angles for each ridgepoint on
               one side of the jet
    
    phi_val2 - 1D array of ridgeline angles for each ridgepoint on
               other side of the jet

    Rlen1 - 1D array of disance from source for each ridgepoint on
            one side of the jet
    
    Rlen2 - 1D array of disance from source for each ridgepoint on
            other side of the jet
    
    Constants
    ---------

    Returns
    -----------
    section_parameters1 - 2D array, shape(n,12)
                          Array with section points (x,y * 4), distance from source,
                          flux, volume and area for one side of the jet

    section_parameters2 - 2D array, shape(n,12)
                          Array with section points (x,y * 4), distance from source
                          flux, volume and area for other side of the jet

    Notes
    -----------
    """

    print(' ')
    print('Starting jet sections processing')
    print('-------------------------------------------')
    start_time = time.time()
    
    # Set up the directory structure for jet sections processing
    JSS.Setup()

    section_parameters1, section_parameters2 = GetEdgepointsAndSections(flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2)

    print('Time taken for jet sections processing = ' + str((time.time()-start_time)/60),'m')

    return section_parameters1, section_parameters2

#############################################

def CommonJetSections():

    """
    Find a common set of jet sections across all maps.

    Returns
    -----------
    start_time - start time for spectral index processing

    common_section_params1 - 3D array, shape(m,n,11)
                             Array with section points (x,y * 4), distance from source,
                             flux and area for one side of the jet.
                             n: number of sections
                             m: number of maps

    common_section_params2 - 3D array, shape(m,n,11)
                             Array with section points (x,y * 4), distance from source
                             flux and area for other side of the jet.
                             n: number of sections
                             m: number of maps

    Notes
    -----------
    """

    print(' ')
    print('Starting spectral index processing')
    print('-------------------------------------------')
    start_time = time.time()

    # Set up the directory structure for spectral index processing
    JSS.Setup_spectral_index()

    common_section_params1, common_section_params2 = CommonJetSectionsAcrossMultipleMaps()

    return start_time, common_section_params1, common_section_params2