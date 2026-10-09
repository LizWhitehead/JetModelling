#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSSetup.py
Sets up the directory structure for JetSection processing.
Created by LizWhitehead - 01/05/2025
"""

import JetModelling_MapSetup as JMS
import sys
import os

def Setup():

    """
    Sets up the directory structure for JetSection processing.
    """

    if JMS.map_number == 0:             # If the first map in this run, clear the section/region folders
        # Initialise required directories under working directory. 
        newdirs = ['sections', 'regions']
        path = os.getcwd()
        for d in newdirs:
            newd = path + '/' + d
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

        # Delete the spectral index sub-directory and contents, if it exists
        newd = path + '/sections/spectralindex'
        if os.path.exists(newd):
            if "win" not in sys.platform.lower():
                os.system("rm -rf " + newd)
            else:
                newd = newd.replace('\\', '/')
                os.system("rmdir /S /Q \"" + newd + "\"")

#############################################

def Setup_spectral_index():

    """
    Sets up the directory structure for spectral index processing.
    Note: the directory will never already exist, as it will have been
    deleted at the start of sections processing.
    """

    path = os.getcwd()
    newd = path + '/sections/spectralindex'
    os.mkdir(newd)
    print ("Made directory ", newd)