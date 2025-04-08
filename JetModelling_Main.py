#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates ridgeline for radio map of single AGN, using the ridgeline toolkit

Usage:  JetRidgeline_Main.py 'FITS file to process' -t 'Map Type [VLA,LOFAR-DR1,LOFAR-DR2,MEERKAT] default=LOFAR-DR2'

Author: LizWhitehead 12/11/2024
"""
import JetRidgeline.Ridgelines as RL
import JetParameters.JetParameters as JP
import argparse
from warnings import simplefilter
simplefilter('ignore') # there is a matplotlib issue with shading on the graphs

# Read in the FITS file and map type arguments
# # parser = argparse.ArgumentParser(description='Create ridgelines for radio map')
# # parser.add_argument(dest='map_file', help='FITS file to process')
# # parser.add_argument('-t', '--Map Type', dest='map_type', default='LOFAR-DR2', choices=['VLA', 'LOFAR-DR1', 'LOFAR-DR2', 'MEERKAT'], help='Default=LOFAR-DR2 (upper case)')
# # args = parser.parse_args()

# # map_type = args.map_type
# # map_file = args.map_file.rstrip()   # Strip off trailing spaces
# # if not map_file.lower().endswith('.fits') : 
# #     map_file = map_file + '.FITS'   # Append the FITS extension if not present
map_file = 'C:/Maps/3C31.HGEOM2Copy.FITS'
map_type = 'VLA'

# Create the jet ridgelines and sections
source_name, section_parameters1, section_parameters2 = RL.CreateRidgelinesAndSections(map_file, map_type)

# Compute parameters along the jet
JP.ComputeJetParameters(map_type, source_name, section_parameters1, section_parameters2)
