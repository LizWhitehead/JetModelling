#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JPSetup.py
Sets up the directory structure for JetParameter processing.
Created by LizWhitehead - 08/04/2025
"""

import JetParameters.JPConstants as JPC
import JetParameters.JPGlobal as JPG
import sys
import os
from astropy.io import fits

def Setup(source_name, map_type):

    """
    Compute jet parameters for each arm of the jet.

    Parameters
    -----------
    source_name - string

    map_type - string

    """
    
    # Initialise constants, specific to the map type
    JPC.init_maptype_specific_constants(map_type)

    # Intialise required directories under working directory. 
    newdirs = ['parameters']
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

    # Set globals
    JPG.sName = source_name
    JPG.sSize = 456                     # source size in pixels (from map)
    JPG.rShift = 0.0169                 # Source red shift
