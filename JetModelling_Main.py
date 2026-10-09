#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetModelling_Main.py
Jet Modelling main program
Created by LizWhitehead - 12/11/2024
"""

if __name__ == '__main__':      # The main function shouldn't execute for spawned parallel processes (use in ridgeline skeletonize)

    import JetModelling_Constants as JMC
    import JetModelling_MapSetup as JMS
    import JetModelling_MapAnalysis as JMA
    import JetRidgeline.Ridgelines as RL
    import JetRidgeline_FromData.Ridgelines_FromData as RLFD
    import JetRidgeline_Skeletonize.Ridgelines_Skeletonize as RLS
    import JetSections.JetSections as JS
    import JetParameters.JetParameters as JP
    from warnings import simplefilter
    simplefilter(action="ignore", category=DeprecationWarning)

    # Populate map data from map file headers
    JMA.PopulateMapData()

    # Loop through all maps included in this run (same AGN, different frequencies)
    for map_number in range(JMS.map_count):

        # Initialise the map
        JMS.InitialiseMap(map_number)

        # Create the jet ridgelines
        if JMC.ridgeline_method == JMC.RidgelineMethod.FROMDATA:
            flux_array = JMA.GetCutout(JMS.map_file, JMS.mCentre, JMS.sSize)    # Create a cutout array, to reduce processing time
            ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2 = RLFD.CreateRidgelines(flux_array)

        elif JMC.ridgeline_method == JMC.RidgelineMethod.RLXID:
            flux_array = JMA.GetCutout(JMS.map_file, JMS.mCentre, JMS.sSize)    # Create a cutout array, to reduce processing time
            ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2 = RL.CreateRidgelines(flux_array)

        else:   # JMC.RidgelineMethod.SKELETONIZE
            flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2 = RLS.CreateRidgelines()

        if not JMC.ridgeline_only:

            # Remove any specified bright sources
            flux_array = JMA.RemoveBrightSources(flux_array, JMS.rmSource)

            # Divide the jet into sections by finding edge points
            section_parameters1, section_parameters2 = JS.CreateJetSections(flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2)

            # Compute parameters along the jet
            JP.ComputeJetParameters(section_parameters1, section_parameters2)

    # if multiple maps for the same AGN, determine the spectral index along the jet
    if JMS.map_count > 1 and not JMC.ridgeline_only:

        # Find the common set of jet sections across all maps
        start_time, common_section_params1, common_section_params2 = JS.CommonJetSections()

        # Determine the spectral index along the jet
        JP.ComputeSpectralIndex(start_time, common_section_params1, common_section_params2)