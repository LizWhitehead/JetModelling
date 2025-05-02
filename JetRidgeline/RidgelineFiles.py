#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:28:40 2020

Filenames for the ridgeline code. This is a template that is completed by the DR2_setup.py script

@author: bonnybarkus ; mangled a bit by JHC
"""
import os

# Load in files -- added by setup script

#indir=os.getenv('RLDIR')      ## Linux
indir = os.getcwd()            ## Windows

LofCat = indir+'/radio.fits'
CompCat = indir+'/components.fits' # changed by MJH -- each dir has
                                   # its own components file now
OptCat = indir+'/optical.fits'
OptCatdf = indir+'/optical.txt'
PossHosts = indir+'/hosts.csv'
tmpdir = indir+'/'

# Data files
fitsfile = indir + '/fits_cutouts/'
npyfile = indir + '/rms_cutouts/'
rms = indir + '/rms/'
rmscutout = indir + '/rms_cutouts/'
fits = indir + '/fits/'
fitscutout = indir + '/fits_cutouts/'

# Ridgelines
TFC = indir+'/total_flux_cutWorkingSet.txt'
Probs = indir+'/problematic/%s_image.png'
R1 = indir+'/ridges/%s_ridge1.txt'
R2 = indir+'/ridges/%s_ridge2.txt'
Rimage = indir+'/ridges/%s_ridges%d.png'
psl = indir+'/problematic/problematic_sources_listWorkingSet.txt'

# SourceSearch
coc = indir+'/CutOutCats/Cutout_Catalogue-%s.txt'
#Dists = 'Catalogue/DistancesFull/distances-%s.txt'
Position = indir+'/Distances/Position_Info.txt'
RDists = indir+'/Distances/Rdistances-%s.txt'
LDists = indir+'/Distances/Ldistances-%s.txt'
NDist = indir+'/Distances/NclosestLdistances-%s.txt'
NLLR = indir+'/Ratios/NearestLofarLikelihoodRatios-%s.txt'
NRLR = indir+'/Ratios/NearestRidgeLikelihoodRatios-%s.txt'
LLR = indir+'/Ratios/LofarLikelihoodRatiosLR-%s.txt'
RLR = indir+'/Ratios/RidgeLikelihoodRatiosLR-%s.txt'
NLRI = indir+'/MagnitudeColour/Nearest30InfoLR-%s.txt'
LRI = indir+'/MagnitudeColour/30InfoLR-%s.txt'
MagCO = indir+'/MagCutOutCats/Cutout_Catalogue-%s.txt'

# Magnitude and Colour Likelihood Ratio
MCLR = indir+'/MagnitudeColour/Nearest30AllLRW1band-%s.txt'
LR = indir+'/MagnitudeColour/AllLRW1bandLR-%s.txt'

