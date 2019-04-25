#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation involving propagation based phase-contrast effect.
We will simulate a phase contrast image of a "marine shell" scanned by a micro-CT scanner.
"""
#%% Imports

from flexdata import geometry    # Geometry definition and display
from flexdata import data        # Convolution operations
from flexdata import display

from flextomo import phantom
from flextomo import projector   # Forward project
from flextomo import model       # Model phase contrast and spectral properties

import numpy

#%% Create volume and forward project:
    
# Initialize images:    
h = 512 # volume size
vol = numpy.zeros([32, h, h], dtype = 'float32')
proj = numpy.zeros([32, 361, h], dtype = 'float32')

# Define a simple projection geometry:
src2obj = 100     # mm
det2obj = 100     # mm   
det_pixel = 0.001 # mm (1 micron)

geom = geometry.circular(src2obj, det2obj, det_pixel, ang_range = [0, 360])

# Create phantom (150 micron wide, 15 micron wall thickness):
vol = phantom.sphere(vol.shape, geom, 0.08)     
vol -= phantom.sphere(vol.shape, geom, 0.07)     

# Project:
projector.forwardproject(proj, vol, geom)

# Show:
display.slice(vol, title = 'Phantom')
display.slice(proj, dim = 0, title = 'Sinogram')

#%% Get the material refraction index of calcium carbonate:

c = model.find_nist_name('Calcium Carbonate')    
rho = c['density'] / 10

energy = 30 # KeV
n = model.material_refraction(energy, 'CaCO3', rho)

#%% Fresnel propagation for phase-contrast:
   
# Create Contrast Transfer Functions for phase contrast effect and detector blurring    
phase_ctf = model.ctf(proj.shape[::2], 'fresnel', [det_pixel, energy, src2obj, det2obj])

sigma = det_pixel 
phase_ctf *= model.ctf(proj.shape[::2], 'gaussian', [det_pixel, sigma * 1])

# Electro-magnetic field image:
proj_i = numpy.exp(-proj * n )

# Field intensity:
data.convolve_filter(proj_i, phase_ctf)
proj_i = numpy.abs(proj_i)**2

display.slice(proj_i, title = 'Sinogram (phase contrast)')

#%% Reconstruct directly:
    
vol_rec = numpy.zeros_like(vol)

projector.FDK(-numpy.log(proj_i), vol_rec, geom)
display.slice(vol_rec, title = 'FDK (Raw)')  
    
#%% Invertion of phase contrast based on dual-CTF model:
    
# Propagator (Dual CTF):
alpha = numpy.imag(n) / numpy.real(n)
dual_ctf = model.ctf(proj.shape[::2], 'dual_ctf', [det_pixel, energy, src2obj, det2obj, alpha])
dual_ctf *= model.ctf(proj.shape[::2], 'gaussian', [det_pixel, sigma])

# Use inverse convolution to solve for blurring and phase contrast
data.deconvolve_filter(proj_i, dual_ctf, epsilon = 0.1)

# Depending on epsilon there is some lof frequency bias introduced...
proj_i /= proj_i.max()

display.slice(proj_i, title = 'Inverted phase contrast')   

# Reconstruct:
vol_rec = numpy.zeros_like(vol)
projector.FDK(-numpy.log(proj_i), vol_rec, geom)
display.slice(vol_rec, title = 'FDK (Dual CTF inversion)')   
