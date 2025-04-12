#!/usr/bin/env python 3
# coding: utf-8

"""
Sets up the directory structure for ridgeline processing. Produces thresholded npy cutout.

Modified by:    LizWhitehead 13/11/2024 to run on a map of single AGN.
"""

import JetRidgeline.RidgelineFiles as RLF
import JetRidgeline.RLConstants as RLC
import JetRidgeline.RLGlobal as RLG
from JetRidgeline.LotssCatalogue.subim import extract_subim
import numpy as np
import sys
import os
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
from os.path import exists

def Setup(map_file, map_type):
    # Sets up the directory structure for ridgeline processing. Produces thresholded npy cutout.
    
    # Initialise constants, specific to the map type
    RLC.init_maptype_specific_constants(map_type)

    # Intialise required directories under working directory. 
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

    # Get source parameters
    hdul = fits.open(map_file)
    hdr = hdul[0].header  # the primary HDU header
    if 'OBJECT' in hdr:
        RLG.sName = str(hdr['OBJECT']).rstrip() # Source name
    else:
        RLG.sName = 'Single_AGN'
    RLG.sRA = float(hdr['CRVAL1'])      # Source RA
    RLG.sDec = float(hdr['CRVAL2'])     # Source Dec
    RLG.sSize = 456                     # source size in pixels (from map)
    RLG.bgRMS = 0.0002                  # rms noise in Jy/beam (from map)
    RLG.rShift = 0.0169                 # Source red shift

    # Create flattened 2D cutout of source. This will work, even if input map file is already 2D.
    flag = get_fits(map_file, RLG.sRA, RLG.sDec, RLG.sName, RLG.sSize*RLC.ddel)     # pass size in degrees

    # Get thresholded npy array
    if flag == 0:
        cutout = str(RLF.fits) + RLG.sName + '.fits'
        nlhdu = fits.open(cutout) 
        d = nlhdu[0].data
        #thres = (1e-3) * RLC.nSig * RLG.bgRMS
        thres = RLC.nSig * RLG.bgRMS

        d[d<thres] = np.nan
        print ("Max val of thresholded array is:", np.nanmax(d))
        np.save(str(RLF.rms) + RLG.sName + '.npy', d)
 
    print ("Completed generating thresholded npy cutout.")


def get_fits(map_file, fra, fdec, fsource, fsize):
    # Create a flattened 2D cutout of the specified size

    sc = SkyCoord(fra*u.deg, fdec*u.deg, frame='icrs')
    s = sc.to_string(style='hmsdms', sep='', precision=2)
    name = fsource
    newsize = 2.5 * fsize

    hdu = extract_subim(map_file, fra, fdec, newsize)
    if hdu is not None:
        hdu.writeto(str(RLF.fits) + name + '.fits', overwrite=True)
        flag = 0
    else:
        print ('Cutout failed for', fsource)
        flag = 1

    return flag

