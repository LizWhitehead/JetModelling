#!/usr/bin/env python 3
# coding: utf-8

"""
RLSSetup.py
Create a flattened 2D version of the input FITS file.
Created by LizWhitehead - 24/08/2025
"""

import JetModelling_MapSetup as JMS
from astropy.io import fits
import os

def Setup():
    if JMS.map_number == 0:             # If the first map in this run, clear the ridgeline folder
        # Initialise ridgeline folders
        newdirs = ['ridges']
        path = os.getcwd()
        for d in newdirs:
            newd=path + '/' + d
            try:
                os.mkdir(newd)
            except:
                # Directory already exists. Empty it.
                print ("Directory", newd, "already exists, cleaning it out")
                newd = newd.replace('\\', '/')
                os.system("del /Q \"" + newd + "\\*\"")
            else:
                # Directory doesn't exist. Create it.
                print ("Made directory ", newd)

    # Get the flux array from the map file
    hdu = fits.open(JMS.map_file)
    flux_array = hdu[0].data.squeeze()
    hdu.close()

    return flux_array