#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load a large CT scan that doesn't fit in computer RAM. Use numpy.memmap to create a disk map.

Change the path variable to your own data path. Change the scrath path to a fast drive (SSD).
Dataset originally used in this example can be downloaded from Zenodo:

https://doi.org/10.5281/zenodo.1144086

"""
#%% Imports

from flexdata import data
from flexdata import display
from flextomo import projector

import numpy

#%% Read data:

path = '/ufs/ciacc/flexbox/good/'
scratch_path = './scratch.mem'

dark = data.read_stack(path, 'di00', sample = 1)
flat = data.read_stack(path, 'io00', sample = 1)
proj = data.read_stack(path, 'scan_', sample = 1, skip = 1, dtype = 'float32', memmap = scratch_path)

geom = data.read_flexraylog(path, sample = 1)

#%% Prepro (use implicit operations):

proj -= dark
proj /= (flat - dark).mean(1)[:, None, :]
numpy.log(proj, out = proj)
proj *= -1

display.slice(proj, title = 'Sinogram. What else?')

#%% FDK Recon:

vol = projector.init_volume(proj)

# Split data into 10 subsets to save memory:
projector.settings.subsets = 10
projector.FDK(proj, vol, geom)

display.slice(vol, bounds = [], title = 'FDK')