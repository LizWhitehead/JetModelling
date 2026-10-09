#!/usr/bin/env python 3
# coding: utf-8

"""
RLFDSetup.py
Initialise ridgeline folders.
Created by LizWhitehead - 21/06/2025
"""

import JetModelling_MapSetup as JMS
import os

def Setup():
    if JMS.map_number == 0:             # If the first map in this run, clear the ridgeline folders

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
