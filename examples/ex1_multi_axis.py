#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test forward / backward projection of a 2D phantom. Various algorithms: EM, SIRT, FISTA. 
Effect of subset version of SIRT.
"""
#%% Imports

from flexdata import geometry
from flexdata import display

from flextomo import projector
from flextomo import phantom

import numpy

#%% Create volume and forward project into two orhogonal scans:
    
# Initialize images:    
proj_a = numpy.zeros([128, 32, 128], dtype = 'float32')
proj_b = numpy.zeros([128, 32, 128], dtype = 'float32')

# Define a simple projection geometry:
geom_a = geometry.circular(src2obj = 100, det2obj = 100, det_pixel = 0.01, ang_range = [0, 360])
geom_b = geom_a.copy()
geom_b['axs_roll'] = 90

# Create phantom and project into proj:
vol = phantom.random_spheroids([128, 128, 128], geom_a, number = 10)
display.slice(vol, title = 'Phantom')

# Forward project:
projector.forwardproject(proj_a, vol, geom_a)
projector.forwardproject(proj_b, vol, geom_b)

display.slice(proj_a, dim = 1, title = 'Proj A')
display.slice(proj_b, dim = 1, title = 'Proj B')

#%% Single-dataset SIRT:

vol_rec = numpy.zeros_like(vol)
projector.SIRT(proj_a, vol_rec, geom_a, iterations = 20)
display.slice(vol_rec, bounds = [0, 1], title = 'SIRT')

#%% Multi-dataset SIRT:

vol_rec = numpy.zeros_like(vol)
projector.SIRT([proj_a, proj_b], vol_rec, [geom_a, geom_b], iterations = 20)
display.slice(vol_rec, bounds = [0, 1], title = 'Multi-data SIRT')
