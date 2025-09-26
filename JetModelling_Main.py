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
    import JetRidgeline.Ridgelines as RL
    import JetRidgeline_FromData.Ridgelines_FromData as RLFD
    import JetRidgeline_Skeletonize.Ridgelines_Skeletonize as RLS
    import JetSections.JetSections as JS
    import JetParameters.JetParameters as JP
    from warnings import simplefilter
    simplefilter('ignore')  # Counteract a matplotlib issue with shading on the graphs

    # Loop through all maps included in this run (same AGN, different frequencies)
    for map_number in range(JMS.map_count):
    #for map_number in range(JMS.map_count-1):
    #for map_number in range(1, JMS.map_count):

        # Initialise the map
        JMS.InitialiseMap(map_number)

        # Create the jet ridgelines
        if JMC.ridgeline_method == JMC.RidgelineMethod.FROMDATA:
            flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2 = RLFD.CreateRidgelines()
        elif JMC.ridgeline_method == JMC.RidgelineMethod.RLXID:
            flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2 = RL.CreateRidgelines()
        else:   # JMC.RidgelineMethod.SKELETONIZE
            flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2 = RLS.CreateRidgelines()

        if not JMC.ridgeline_only:

            # Divide the jet into sections by finding edge points
            section_parameters1, section_parameters2 = JS.CreateJetSections(flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2)

            # Compute parameters along the jet
            JP.ComputeJetParameters(section_parameters1, section_parameters2)
