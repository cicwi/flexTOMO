#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load a standard CT scan that fits in RAM. Reconstruct using different methods.
"""
#%% Imports

from flexdata import data
from flexdata import display
from flextomo import projector

import numpy

#%% Read data:
    
path = 'D:\data\skull'

dark = data.read_stack(path, 'di00', sample = 4)
flat = data.read_stack(path, 'io00', sample = 4)    
proj = data.read_stack(path, 'scan_', sample = 4, skip = 4)

geom = data.read_flexraylog(path)   
 
#%% Prepro:
    
proj = (proj - dark) / (flat.mean(0) - dark)
proj = -numpy.log(proj)
proj = data.flipdim(proj)    

display.slice(proj, title = 'Sinogram. What else?')

#%% FDK Recon

vol = projector.init_volume(proj)
projector.FDK(proj, vol, geom)

display.slice(vol, bounds = [], title = 'FDK')

#%% SIRT with additional options

vol = projector.init_volume(proj)

projector.settings.bounds = [0, 10]
projector.settings.subsets = 10
projector.settings.sorting = 'equidistant'

projector.SIRT(proj, vol, geom, iterations = 3)

display.slice(vol, title = 'SIRT')