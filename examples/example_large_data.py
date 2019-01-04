#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load a large dataset using numpy.memmap - array mapped on disk. Reconstruct it. 
"""
#%% Imports

from flexdata import io
from flexdata import array
from flexdata import display

from flextomo import project

import numpy

#%% Read data

path = '/ufs/ciacc/flexbox/al_test/90KV_no_filt/'

dark = io.read_tiffs(path, 'di00')
flat = io.read_tiffs(path, 'io00')    
proj = io.read_tiffs(path, 'scan_', memmap = 'D:/Data/swap.prj')

meta = io.read_meta(path, 'flexray')   
 
#%% Prepro:
    
# Now, since the data is on the harddisk, we shouldn't lose the pointer to it!    
# Be careful which operations to apply. Implicit are OK.
proj -= dark
proj /= (flat.mean(0) - dark)

numpy.log(proj, out = proj)
proj *= -1

proj = array.raw2astra(proj)    

display.slice(proj)

#%% Recon

vol = numpy.zeros([50, 1000, 1000], dtype = 'float32')

# Split the data into 20 subsets:
project.settings['block_number'] = 20

project.FDK(proj, vol, meta['geometry'])

display.slice(vol)

#%% SIRT

vol = numpy.ones([50, 1000, 1000], dtype = 'float32')

project.settings['bounds'] = [0, 10]
project.settings['mode'] = 'equidistant'

project.SIRT(proj, vol, meta['geometry'], iterations = 5)

display.slice(vol, title = 'SIRT')