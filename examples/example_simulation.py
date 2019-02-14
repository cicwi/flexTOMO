#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test forward / backward projection of a 2D phantom. Various algorithms: EM, SIRT, FISTA. 
Effect of subset version of SIRT.
"""
#%% Imports

from flexdata import geometry
from flexdata import display

from flextomo import project
from flextomo import phantom

import numpy

#%% Create volume and forward project:
    
# Initialize images:    
proj = numpy.zeros([1, 128, 512], dtype = 'float32')

# Define a simple projection geometry:
geom = geometry.basic(src2obj = 100, det2obj = 100, det_pixel = 0.01, ang_range = [0, 360])

print('Volume width is:', 512 * geom.voxel)

# Create phantom and project into proj:
vol = phantom.abstract_nudes([1, 512, 512], geom, complexity = 10)

display.slice(vol, title = 'Phantom')

# Forward project:
project.forwardproject(proj, vol, geom)
display.slice(proj, title = 'Sinogram')

#%% Unfiltered back-project

# Make volume:
vol_rec = numpy.zeros_like(vol)

# Backproject:
project.settings['block_number'] = 1
project.backproject(proj, vol_rec, geom)

display.slice(vol_rec, title = 'Backprojection')

#%% Reconstruct

# Make volume:
vol_rec = numpy.zeros_like(vol)

# Use FDK:
project.FDK(proj, vol_rec, geom)

display.slice(vol_rec, title = 'FDK')

#%% Use Expectation Maximization:

vol_rec = numpy.zeros_like(vol)

project.EM(proj, vol_rec, geom, iterations = 10)

display.slice(vol_rec, title = 'EM')

#%% SIRT

vol = numpy.zeros([1, 512, 512], dtype = 'float32')

project.SIRT(proj, vol, geom, iterations = 10)

display.slice(vol, title = 'SIRT')

#%% FISTA

vol = numpy.zeros([1, 512, 512], dtype = 'float32')

project.FISTA(proj, vol, geom, iterations = 10)

display.slice(vol, title = 'FISTA')

#%% SIRT with Subsets and non-negativity:

project.settings['bounds'] = [0, 10]
project.settings['block_number'] = 10
project.settings['mode'] = 'equidistant'

vol = numpy.zeros([1, 512, 512], dtype = 'float32')

project.SIRT(proj, vol, geom, iterations = 10)

display.slice(vol, title = 'SIRT')


