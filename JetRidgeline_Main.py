#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates ridgelines for radio map using the ridgeline toolkit

Usage:  JetRidgeline_Main.py 'FITS file to process' -t 'Map Type [VLA,LOFAR-DR1,LOFAR-DR2,MEERKAT] default=LOFAR-DR2'

Author: LizWhitehead 12/11/2024
"""
import JetRidgeline.RLConstants as RLC
import argparse

# Read in the FITS file and map type arguments
# # parser = argparse.ArgumentParser(description='Create ridgelines for radio map')
# # parser.add_argument(dest='fits_file', help='FITS file to process')
# # parser.add_argument('-t', '--Map Type', dest='map_type', default='LOFAR-DR2', choices=['VLA', 'LOFAR-DR1', 'LOFAR-DR2', 'MEERKAT'], help='Default=LOFAR-DR2 (upper case)')
# # args = parser.parse_args()

# # map_type = args.map_type
# # fits_file = args.fits_file.rstrip()   # Strip off trailing spaces
# # if fits_file.lower() != '.fits': 
# #     fits_file = fits_file + '.FITS'   # Append the FITS extension if not present
fits_file = '3C31.HGEOM2 - Copy.FITS'
map_type = 'VLA'

# Initialise constants, specific to the map type
RLC.init_maptype_specific_constants(map_type)
print (RLC.R); print(RLC.rdel); print(RLC.ddel); print(RLC.nSig)

##flux_for_files(args.files,args.fgr,args.bgr,args.indiv,args.bgsub,verbose=args.verbose)
