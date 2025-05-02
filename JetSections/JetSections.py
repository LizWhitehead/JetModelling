#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetSections.py
Divide both arms of the jet into sections.
Created by LizWhitehead - 30/04/2025
"""

import JetSections.JSSetup as JSS
import JetSections.JetSectionToolkit as JSTK
import JetSections.JSConstants as JSC
import JetRidgeline.RLConstants as RLC
import JetModelling_MapSetup as JMS
import numpy as np

def CreateJetSections(area_fluxes, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2):

    """
    Divide both arms of the jet into sections by finding edge points.

    Parameters
    -----------
    area_fluxes - 2D array,
                  raw image array

    ridge1 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for one arm of the jet
    
    ridge2 - 2D array, shape(n,2)
             Array of ridgepoint co-ordinates for other arm of the jet

    phi_val1 - 1D array of ridgeline angles for each ridgepoint on
               one arm of the jet
    
    phi_val2 - 1D array of ridgeline angles for each ridgepoint on
               other arm of the jet

    Rlen1 - 1D array of disance from source for each ridgepoint on
            one arm of the jet
    
    Rlen2 - 1D array of disance from source for each ridgepoint on
            other arm of the jet
    
    Constants
    ---------

    Returns
    -----------
    section_parameters1 - 2D array, shape(n,12)
                          Array with section points (x,y * 4), distance from source,
                          flux and volume for one arm of the jet

    section_parameters2 - 2D array, shape(n,12)
                          Array with section points (x,y * 4), distance from source
                          flux and volume for other arm of the jet

    Notes
    -----------
    """
    
    # Set up the directory structure for jet sections processing
    JSS.Setup()

    edge_points1 = edge_points2 = np.empty((0,5))    # Initialise edge point arrays
    section_parameters1 = section_parameters2 = np.empty((0,12))

    # Loop through the ridge points and find the corresponding edgepoints
    if not np.all(np.isnan(ridge1)) and not np.all(np.isnan(ridge2)):

        print('Finding edge points')
    
        # Initialise flags to indicate that edge points determination should stop
        stop_finding_edge_points1 = False; stop_finding_edge_points2 = False

        # Find initial edge points
        init_edge_points = JSTK.FindInitEdgePoints(area_fluxes, np.array(ridge1[0]), phi_val1[0], Rlen1[0])
        edge_points1 = np.vstack((edge_points1, init_edge_points)); edge_points2 = np.vstack((edge_points2, init_edge_points))

        # Find edgepoints for one arm of the jet
        ridge_count = 1; ridge_total = len(ridge1)
        while ridge_count < ridge_total:

            if np.isnan(phi_val1[ridge_count]):
                break

            if (Rlen1[ridge_count] - Rlen1[ridge_count-1]) < (JSC.MaxRFactor * RLC.R):  # Test whether the last step size has increased by too much
                stop_finding_edge_points1, new_edge_points1 = \
                    JSTK.FindEdgePoints(area_fluxes, ridge1[ridge_count], phi_val1[ridge_count], Rlen1[ridge_count], prev_edge_points = edge_points1[-1])
            else:
                new_edge_points1 = JSTK.FindInitEdgePoints(area_fluxes, ridge1[ridge_count], phi_val1[ridge_count], Rlen1[ridge_count])  # Re-initialise edge points algorithm

            if not np.isnan(new_edge_points1).any(): 
                edge_points1 = np.vstack((edge_points1, new_edge_points1))

            ridge_count += 1

        # Find edgepoints for other arm of the jet
        ridge_count = 1; ridge_total = len(ridge2)
        while ridge_count < ridge_total:

            if np.isnan(phi_val2[ridge_count]):
                break

            if (Rlen2[ridge_count] - Rlen2[ridge_count-1]) < (JSC.MaxRFactor * RLC.R):  # Test whether the last step size has increased by too much
                stop_finding_edge_points2, new_edge_points2 = \
                    JSTK.FindEdgePoints(area_fluxes, ridge2[ridge_count], phi_val2[ridge_count], Rlen2[ridge_count], prev_edge_points = edge_points2[-1])
            else:
                new_edge_points2 = JSTK.FindInitEdgePoints(area_fluxes, ridge2[ridge_count], phi_val2[ridge_count], Rlen2[ridge_count])  # Re-initialise edge points algorithm

            if not np.isnan(new_edge_points2).any(): 
                edge_points2 = np.vstack((edge_points2, new_edge_points2))

            ridge_count += 1

        # Interpolate extra edge points at points in the jet where significant flux is cut off
        print('Adding extra edge points')
        edge_points1 = JSTK.AddEdgePoints(area_fluxes, edge_points1)
        edge_points2 = JSTK.AddEdgePoints(area_fluxes, edge_points2)

        # Get sections and section parameters (distance from source, flux, volume) of the jet
        print('Getting jet sections')
        section_parameters1, section_parameters2 = JSTK.GetJetSections(area_fluxes, edge_points1, edge_points2)

        # Save files and plot data
        JSTK.SaveEdgepointFiles(JMS.sName, edge_points1, edge_points2, section_parameters1, section_parameters2)
        JSTK.PlotEdgePoints(area_fluxes, JMS.sName, edge_points1, edge_points2, section_parameters1, section_parameters2)

    return section_parameters1, section_parameters2
