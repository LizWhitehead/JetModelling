#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates ridgeline for radio map of single AGN, using the ridgeline toolkit

Usage:  JetRidgeline_Main.py 'FITS file to process' -t 'Map Type [VLA,LOFAR-DR1,LOFAR-DR2,MEERKAT] default=LOFAR-DR2'

Author: LizWhitehead 12/11/2024
"""
import JetModelling_MapSetup as JMS
import JetRidgeline.Ridgelines as RL
import JetParameters.JetParameters as JP
from warnings import simplefilter
simplefilter('ignore') # there is a matplotlib issue with shading on the graphs

JMS.setup_map_specific_parameters()

# Create the jet ridgelines and sections
section_parameters1, section_parameters2 = RL.CreateRidgelinesAndSections()

# Compute parameters along the jet
JP.ComputeJetParameters(section_parameters1, section_parameters2)
