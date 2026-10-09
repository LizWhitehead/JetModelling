#!/usr/bin/env python 3
# coding: utf-8

"""
RLSetup.py
Initialise ridgeline folders.
Modified by:    LizWhitehead 13/11/2024
"""

import JetModelling_MapSetup as JMS
import sys
import os

def Setup():
    if JMS.map_number == 0:             # If the first map in this run, clear the ridgeline folder

        # Initialise ridgeline folders 
        newdirs = ['fits','rms','fits_cutouts','rms_cutouts','Distances','MagnitudeColour','Ratios','CutOutCats','MagCutOutCats','badsources_output','ridges','problematic','cutouts']
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

