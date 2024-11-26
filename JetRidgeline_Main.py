#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates ridgeline for radio map of single AGN, using the ridgeline toolkit

Usage:  JetRidgeline_Main.py 'FITS file to process' -t 'Map Type [VLA,LOFAR-DR1,LOFAR-DR2,MEERKAT] default=LOFAR-DR2'

Author: LizWhitehead 12/11/2024
"""
import JetRidgeline.RLConstants as RLC
import JetRidgeline.RLSetup as RLS
import JetRidgeline.Ridgelines as RL
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

# Set up the directory structure for ridgeline processing. Produce 2D flattened map and thresholded npy cutout.
RLS.Setup(map_file, map_type)

# Draw the ridgelines
RL.Draw_Ridgelines()