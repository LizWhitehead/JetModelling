#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JPSetup.py
Sets up the directory structure for JetParameter processing.
Created by LizWhitehead - 08/04/2025
"""

import JetModelling_MapSetup as JMS
import sys
import os

def Setup():
    """
    # Sets up the directory structure for JetParameter processing.
    """

    if JMS.map_number == 0:             # If the first map in this run, clear the parameter folder
        # Initialise required directories under working directory. 
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
