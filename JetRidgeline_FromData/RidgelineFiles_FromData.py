#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RidgelineFiles_FromData.py
Define folder and file names.
Created by LizWhitehead - 21/06/2025
"""

import os

# Load in files -- added by setup script

indir = os.getcwd()

# Ridgelines
R1 = indir + '/ridges/%s_ridge1.txt'
R2 = indir + '/ridges/%s_ridge2.txt'
Rimage = indir + '/ridges/%s_ridges.png'
