#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load a standard CT scan. Reconstruct using different methods.
"""
#%% Imports

from flexdata import io
from flexdata import array
from flexdata import display

from flextomo import project

import numpy

#%% Read data:
    
path = '/ufs/ciacc/flexbox/al_test/90KV_no_filt/'

dark = io.read_tiffs(path, 'di00')
flat = io.read_tiffs(path, 'io00')    
proj = io.read_tiffs(path, 'scan_')

meta = io.read_meta(path, 'flexray')   
 
#%% Prepro:
    
proj = (proj - dark) / (flat.mean(0) - dark)
proj = -numpy.log(proj)

proj = array.raw2astra(proj)    

display.display_slice(proj, title = 'Sinogram. What else?')

#%% FDK Recon

vol = project.init_volume(proj)
project.FDK(proj, vol, meta['geometry'])

display.display_slice(vol, bounds = [], title = 'FDK')

#%% EM

vol = numpy.ones([10, 2000, 2000], dtype = 'float32')

project.EM(proj, vol, meta['geometry'], iterations = 3)

display.display_slice(vol, title = 'EM')

#%% SIRT with additional options

vol = numpy.zeros([1, 2000, 2000], dtype = 'float32')

project.settings['bounds'] = [0, 10]
project.settings['block_number'] = 20
project.settings['mode'] = 'equidistant'

project.SIRT(proj, vol, meta['geometry'], iterations = 3)

display.display_slice(vol, title = 'SIRT')