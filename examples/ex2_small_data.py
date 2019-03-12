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
    
path = '/ufs/ciacc/flexbox/good/'

dark = data.read_stack(path, 'di00', sample = 2)
flat = data.read_stack(path, 'io00', sample = 2)    
proj = data.read_stack(path, 'scan_', sample = 2, skip = 2)

geom = data.read_flexraylog(path, sample = 2)   
 
#%% Prepro:
    
flat = (flat - dark).mean(1)
proj = (proj - dark) / flat[:, None, :]
proj = -numpy.log(proj).astype('float32')

display.slice(proj, dim = 1, title = 'Projection')

#%% FDK Recon

vol = projector.init_volume(proj)
projector.FDK(proj, vol, geom)

display.slice(vol, bounds = [], title = 'FDK')

#%% SIRT with additional options

vol = projector.init_volume(proj)

projector.settings.bounds = [0, 10]
projector.settings.subsets = 10
projector.settings.sorting = 'equidistant'

projector.SIRT(proj, vol, geom, iterations = 5)

display.slice(vol, title = 'SIRT')