#!/usr/bin/env python 3
# coding: utf-8

"""
RLFDSetup.py
Create a flattened 2D version of the input FITS file.
Created by LizWhitehead - 21/06/2025
"""

import JetRidgeline_FromData.RidgelineFiles_FromData as RLFDF
import JetModelling_MapSetup as JMS
from JetRidgeline_FromData.LotssCatalogue.subim import extract_subim
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
import os

def Setup():
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

    return flux_array