#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetParameterFiles.py
Filenames for the jet parameter modelling code.
Created by LizWhitehead - 08/04/2025
"""

import os

indir = os.getcwd()            # Windows

# Jet parameters files
JP_kappa0_side1             = indir + '/parameters/%s_%s_kappa0_side1.txt'
JP_kappa0_side2             = indir + '/parameters/%s_%s_kappa0_side2.txt'
JP_kappa0_plot_pressure     = indir + '/parameters/%s_%s_kappa0_pressure.png'
JP_flux_area_plot           = indir + '/parameters/%s_%s_flux_area.png'

# Spectral index files
JP_spindex_fluxarea_side1   = indir + '/parameters/spectralindex/%s_%s_fluxarea_side1.txt'
JP_spindex_fluxarea_side2   = indir + '/parameters/spectralindex/%s_%s_fluxarea_side2.txt'
JP_spindex_fluxarea_plot    = indir + '/parameters/spectralindex/%s_%s_flux_area.png'
JP_spindex_data_side1       = indir + '/parameters/spectralindex/%s_spectralindex_data_side1.txt'
JP_spindex_data_side2       = indir + '/parameters/spectralindex/%s_spectralindex_data_side2.txt'
JP_spectralindex_side1      = indir + '/parameters/spectralindex/%s_spectralindex_side1.txt'
JP_spectralindex_side2      = indir + '/parameters/spectralindex/%s_spectralindex_side2.txt'
JP_spectralindex_plot_side1 = indir + '/parameters/spectralindex/%s_spectralindex_side1 (%s to %s kpc).png'
JP_spectralindex_plot_side2 = indir + '/parameters/spectralindex/%s_spectralindex_side2 (%s to %s kpc).png'
