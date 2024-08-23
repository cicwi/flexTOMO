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

#%% Create volume and forward project (32 projections):

# Initialize images:
proj = numpy.zeros([30, 64, 256], dtype = 'float32')

# Define a simple projection geometry:
geom = geometry.circular(src2obj = 100, det2obj = 100, det_pixel = 0.01, ang_range = [0, 360])

# Create phantom and project into proj:
vol = phantom.abstract_nudes([30, 256, 256], geom, complexity = 8)
display.slice(vol, title = 'Phantom')

# Forward project:
projector.forwardproject(proj, vol, geom)
display.slice(proj, title = 'Sinogram')

#%% Unfiltered back-project

# Make volume:
vol_rec = numpy.zeros_like(vol)

# Backproject:
projector.backproject(proj, vol_rec, geom)
display.slice(vol_rec, title = 'Backprojection')

#%% Filtered back-project

# Make volume:
vol_rec = numpy.zeros_like(vol)

# Use FDK:
projector.FDK(proj, vol_rec, geom)
display.slice(vol_rec, title = 'FDK')

#%% Simple SIRT:

vol_rec = numpy.zeros_like(vol)
projector.SIRT(proj, vol_rec, geom, iterations = 20)
display.slice(vol_rec, title = 'SIRT')

#%% SIRT with subsets and non-negativity:

# Settings:
projector.settings.update_residual = True
projector.settings.bounds = [0, 2]
projector.settings.subsets = 10
projector.settings.sorting = 'equidistant'

# Reonstruction:
vol_rec = numpy.zeros_like(vol)
projector.SIRT(proj, vol_rec, geom, iterations = 20)
display.slice(vol_rec, title = 'SIRT')

#%% FISTA

vol_rec = numpy.zeros_like(vol)
projector.settings.subsets = 10
projector.FISTA(proj, vol_rec, geom, iterations = 20, lmbda = 1e-3)
display.slice(vol_rec, title = 'FISTA')

#%% STUDENTS-T

vol_rec = numpy.zeros_like(vol)
projector.settings.subsets = 10
projector.settings.student = True
projector.SIRT(proj, vol_rec, geom, iterations = 20)
display.slice(vol_rec, title = 'StudentsT')

#%% Expectation Maximization:

vol_rec = numpy.ones_like(vol)
projector.EM(proj, vol_rec, geom, iterations = 20)
display.slice(vol_rec, bounds = [0, 1], title = 'EM')

#%% PWLS:

vol_rec = numpy.zeros_like(vol)
projector.PWLS(proj, vol_rec, geom, iterations = 20)
display.slice(vol_rec, bounds = [0, 1], title = 'PWLS')