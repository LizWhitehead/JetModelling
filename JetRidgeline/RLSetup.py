#!/usr/bin/env python 3
# coding: utf-8

"""
Sets up the directory structure for ridgeline processing. Produces thresholded npy cutout.

Modified by:    LizWhitehead 13/11/2024 to run on a map of single AGN.
"""

import JetModelling_MapSetup as JMS
import JetRidgeline.RidgelineFiles as RLF
import JetRidgeline.RLConstants as RLC
from JetRidgeline.LotssCatalogue.subim import extract_subim
import numpy as np
import sys
import os
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
from os.path import exists

def Setup():
    # Sets up the directory structure for ridgeline processing. Produces thresholded npy cutout.

    # Initialise required directories under working directory. 
    newdirs = ['fits','rms','fits_cutouts','rms_cutouts','Distances','MagnitudeColour','Ratios','CutOutCats','MagCutOutCats','badsources_output','ridges','edgepoints','problematic','cutouts']
    path = os.getcwd()
    for d in newdirs:
        newd=path + '/' + d
        try:
            os.mkdir(newd)
        except:
            # Directory already exists. Empty it.
            print ("Directory", newd, "already exists, cleaning it out")
            if "win" not in sys.platform.lower():
                os.system("rm " + newd + "/*")
            else:
                newd = newd.replace('\\', '/')
                os.system("del /Q \"" + newd + "\\*\"")
        else:
            # Directory doesn't exist. Create it.
            print ("Made directory ", newd)

    # Create flattened 2D cutout of source. This will work, even if input map file is already 2D.
    flag = get_fits(JMS.sRA, JMS.sDec, JMS.sName, JMS.sSize*JMS.ddel)     # pass size in degrees

    # Get thresholded npy array
    if flag == 0:
        cutout = str(RLF.fits) + JMS.sName + '.fits'
        nlhdu = fits.open(cutout) 
        d = nlhdu[0].data
        #thres = (1e-3) * JMS.nSig * JMS.bgRMS
        thres = JMS.nSig * JMS.bgRMS

        d[d<thres] = np.nan
        print ("Max val of thresholded array is:", np.nanmax(d))
        np.save(str(RLF.rms) + JMS.sName + '.npy', d)
 
    print ("Completed generating thresholded npy cutout.")


def get_fits(fra, fdec, fsource, fsize):
    # Create a flattened 2D cutout of the specified size

    sc = SkyCoord(fra*u.deg, fdec*u.deg, frame='icrs')
    s = sc.to_string(style='hmsdms', sep='', precision=2)
    name = fsource
    newsize = 2.5 * fsize

    hdu = extract_subim(JMS.map_file, fra, fdec, newsize)
    if hdu is not None:
        hdu.writeto(str(RLF.fits) + name + '.fits', overwrite=True)
        flag = 0
    else:
        print ('Cutout failed for', fsource)
        flag = 1

    return flag

