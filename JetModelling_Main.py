#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates ridgeline for radio map of single AGN, using the ridgeline toolkit

Usage:  JetRidgeline_Main.py 'FITS file to process' -t 'Map Type [VLA,LOFAR-DR1,LOFAR-DR2,MEERKAT] default=LOFAR-DR2'

Author: LizWhitehead 12/11/2024
"""
import JetModelling_Constants as JMC
import JetRidgeline.Ridgelines as RL
import JetRidgeline_FromData.Ridgelines_FromData as RLFD
import JetSections.JetSections as JS
import JetParameters.JetParameters as JP
from warnings import simplefilter
simplefilter('ignore') # there is a matplotlib issue with shading on the graphs

# Create the jet ridgelines
if JMC.ridgelines_from_data:
    flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2 = RLFD.CreateRidgelines()
else:
    flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2 = RL.CreateRidgelines()

if not JMC.ridgeline_only:

    # Divide the jet into sections by finding edge points
    section_parameters1, section_parameters2 = JS.CreateJetSections(flux_array, ridge1, phi_val1, Rlen1, ridge2, phi_val2, Rlen2)

    # Compute parameters along the jet
    JP.ComputeJetParameters(section_parameters1, section_parameters2)
