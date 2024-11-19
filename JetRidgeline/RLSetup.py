#!/usr/bin/env python 3
# coding: utf-8

"""
Sets up the directory structure for ridgeline processing. Produces thresholded npy cutout.

Modified by:    LizWhitehead 13/11/2024 to run on a map of single AGN.
"""

import JetRidgeline.RLConstants as RLC
from JetRidgeline.subim import extract_subim
import numpy as np
import sys
import os
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
from os.path import exists
from warnings import simplefilter
simplefilter('ignore') # there is a matplotlib issue with shading on the graphs

def setup(map_file, map_type):
    # Sets up the directory structure for ridgeline processing. Produces thresholded npy cutout.
    
    # Initialise constants, specific to the map type
    RLC.init_maptype_specific_constants(map_type)
    print (RLC.R); print(RLC.rdel); print(RLC.ddel); print(RLC.nSig)
    print (map_file)
    print (map_type)

    # Intialise required directories under working directory. 
    newdirs = ['fits','rms4','fits_cutouts','rms4_cutouts','Distances','MagnitudeColour','Ratios','CutOutCats','MagCutOutCats','badsources_output','ridges','problematic','cutouts']
    path = os.getcwd()
    for d in newdirs:
        newd=path + '/' + d
        try:
            os.mkdir(newd)
        except:
            # Directory already exists. Empty it.
            print ("Directory", newd, "already exists, cleaning it out")
            os.system("rm " + newd + "/*")
        else:
            # Directory doesn't exist. Create it.
            print ("Made directory ", newd)

    # Extract cutout and thresholded npy array

    # Get values from the map file header
    hdul = fits.open(map_file)
    hdr = hdul[0].header  # the primary HDU header
    if 'OBJECT' in hdr:
        ssource = str(hdr['OBJECT']).rstrip()   # Source name
    else:
        ssource = 'Single_AGN'
    sra = float(hdr['CRVAL1'])              # Source RA
    sdec = float(hdr['CRVAL2'])             # Source Dec
    ##LW##flux = row['Peak_flux']
    rms = 0.0002                            # rms noise in Jy/beam (from map)
    ssize = 456 * RLC.ddel * 3600           # source size in arcsecs (from map)

    # Create flattened 2D cutout of source. This will work, even if input map file is already 2D.
    flag = get_fits(map_file, sra, sdec, ssource, ssize)

    # Get thresholded npy array
    if flag == 0:
        cutout=path+'/fits/'+ssource+'.fits'
        nlhdu=fits.open(cutout) 
        d=nlhdu[0].data
        thres = (1e-3) * RLC.nSig * rms     ##LW## Why does it times by 1e-3

        d[d<thres] = np.nan
        mtest = np.nanmax(d)
        print ("Max val of thresholded array is:", mtest)
        np.save(path + "/rms4/" + ssource + '.npy', d)
 
    print ("Completed generating thresholded npy cutout.")

    # Append input and output lines to RidgelineFiles template
    '''
    rlines=[l.rstrip().split(",") for l in open(inridge).readlines()]

    rfile=open(inridge,"a")

    rfile.write("LofCat = \""+sourcecat+"\"\n")
    rfile.write("CompCat = \""+compcat+"\"\n")
    rfile.write("OptCat = \""+hostcat+"\"\n")
    rfile.write("PossHosts = \""+outroot+"_RLhosts.csv\"\n")

    rfile.close()

    cpcmd="cp "+sourcecat+" radio.fits"
    os.system(cpcmd)
    '''

def get_fits(map_file, fra, fdec, fsource, fsize):
    # Create a flattened 2D cutout of the specified size

    sc = SkyCoord(fra*u.deg, fdec*u.deg, frame='icrs')
    s = sc.to_string(style='hmsdms', sep='', precision=2)
    name = fsource
    newsize = 2.5 * fsize / 3600.0

    hdu = extract_subim(map_file, fra, fdec, newsize)
    if hdu is not None:
        hdu.writeto('fits/' + name + '.fits', overwrite=True)
        flag = 0
    else:
        print ('Cutout failed for', fsource)
        flag = 1

    return flag

