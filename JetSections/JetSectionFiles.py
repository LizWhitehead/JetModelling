#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetSectionFiles.py
Filenames for the jet sections code.
Created by LizWhitehead - 01/05/2025
"""

import os

indir = os.getcwd()            # Windows

# Jet sections files
EP1 = indir+'/sections/%s_edgepoint1.txt'
EP2 = indir+'/sections/%s_edgepoint2.txt'
SP1 = indir+'/sections/%s_sectionparameters1.txt'
SP2 = indir+'/sections/%s_sectionparameters2.txt'
SR1 = indir+'/regions/%s_sectionregions1.txt'
SR2 = indir+'/regions/%s_sectionregions2.txt'
RGS = indir+'/regions/%s_all_sections.reg'
EPimage = indir+'/sections/%s_edgepoints.png'
SCimage = indir+'/sections/%s_sections.png'
