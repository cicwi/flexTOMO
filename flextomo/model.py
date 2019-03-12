#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kostenko

This module can be used to simulate spectral effects and resolution/photon-count effects.
NIST data is used (embedded in xraylib module) to simulate x-ray spectra of compounds
"""
# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Imports >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import numpy
import xraylib
import warnings

from flextomo import projector
from flexdata import display

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Constants >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Some useful physical constants:
phys_const = {'c': 299792458, 'h': 6.62606896e-34, 'h_ev': 4.13566733e-15, 'h_bar': 1.054571628e-34, 'h_bar_ev': 6.58211899e-16,
              'e': 1.602176565e-19, 'Na': 6.02214179e23, 're': 2.817940289458e-15, 'me': 9.10938215e-31, 'me_ev': 0.510998910e6}
const_unit = {'c': 'm/c', 'h': 'J*S', 'h_ev': 'e*Vs', 'h_bar': 'J*s', 'h_bar_ev': 'eV*s',
              'e': 'colomb', 'Na': '1/mol', 're': 'm', 'me': 'kg', 'me_ev': 'ev/c**2'}

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def ctf(shape, mode = 'gaussian', parameter = (1, 1)):
    """
    Get a CTF (fft2(PSF)) of one of the following types: gaussian, dual_ctf, fresnel, TIE
    
    Args:
        shape (list): shape of a projection image
        mode (str): 'gaussian' (blurr), 'dual_ctf', fresnel, TIE (phase contrast)
        parameter (list / float): PSF/CTF parameters. 
                  For gaussian: [detector_pixel, sigma]
                  For dual_ctf/tie: [detector_pixel, energy, src2obj, det2obj, alpha]
                  For fresnel: [detector_pixel, energy, src2obj, det2obj]
    """
    # Some constants...    
    h_bar_ev = 6.58211899e-16
    c= 299792458

    if mode == 'gaussian':
        
        # Gaussian CTF:
        pixel = parameter[0]
        sigma = parameter[1]
          
        u = _w_space_(shape, 0, pixel)
        v = _w_space_(shape, 1, pixel)
        
        ctf = numpy.exp(-((u[:, None] * sigma) ** 2 + (v[None, :] * sigma) ** 2)/2)
        
    elif mode == 'dual_ctf':
        
        # Dual CTF approximation phase contrast propagator:
        pixelsize = parameter[0]
        energy = parameter[1]
        r1 = parameter[2]
        r2 = parameter[3]
        alpha = parameter[4]
        
        # Effective propagation distance:
        m = (r1 + r2) / r1
        r_eff = r2 / m
        
        # Wavenumber;
        k = energy / (h_bar_ev * c)
        
        # Frequency square:
        w2 = _w2_space_(shape, pixelsize)
        
        #return -2 * numpy.cos(w2 * r_eff / (2*k)) + 2 * (alpha) * numpy.sin(w2 * r_eff / (2*k))
        ctf = numpy.cos(w2 * r_eff / (2*k)) - (alpha) * numpy.sin(w2 * r_eff / (2*k))
    
    elif mode == 'fresnel':
        
        # Fresnel propagator for phase contrast simulation:
        pixelsize = parameter[0]
        energy = parameter[1]
        r1 = parameter[2]
        r2 = parameter[3]
        
        # Effective propagation distance:
        m = (r1 + r2) / r1
        r_eff = r2 / m
        
        # Wavenumber;
        k = energy / (h_bar_ev * c)
        
        # Frequency square:
        w2 = _w2_space_(shape, pixelsize)
        
        ctf = numpy.exp(1j * w2 * r_eff / (2*k))
        
    elif mode == 'tie':
        
        # Transport of intensity equation approximation of phase contrast:
        pixelsize = parameter[0]
        energy = parameter[1]
        r1 = parameter[2]
        r2 = parameter[3]
        alpha = parameter[4]
        
        # Effective propagation distance:
        m = (r1 + r2) / r1
        r_eff = r2 / m
        
        # Wavenumber;
        k = energy / (h_bar_ev * c)
        
        # Frequency square:
        w2 = _w2_space_(shape, pixelsize)
        
        ctf = 1 - alpha * w2 * r_eff / (2*k)
       
    return numpy.fft.fftshift(ctf)
    
def _w_space_(shape, dim, pixelsize):
    """
    Generate spatial frequencies along dimension dim.
    """                   
    # Image dimentions:
    sz = numpy.array(shape) * pixelsize
        
    # Frequency:
    xx = numpy.arange(0, shape[dim]) - shape[dim]//2
    return 2 * numpy.pi * xx / sz[dim]

def _w2_space_(shape, pixelsize):
    """
    Generates the lambda*freq**2*R image that can be used to calculate phase propagator at distance R, photon wavelength lambda.
    """
    # Frequency squared:
    u = _w_space_(shape, 0, pixelsize)
    v = _w_space_(shape, 1, pixelsize)
    return (u**2)[:, None] + (v**2)[None, :]
            
def apply_noise(image, mode = 'poisson', parameter = 1):
    """
    Add noise to the data.
    
    Args:
        image (numpy.array): image to apply noise to
        mode (str): poisson or normal
        parameter (float): norm factor for poisson or a standard deviation    
    """
    
    if mode == 'poisson':
        return numpy.random.poisson(image * parameter)
        
    elif mode == 'normal':
        return numpy.random.normal(image, parameter)
        
    else: 
        raise ValueError('Me not recognize the mode! Use normal or poisson!')

def effective_spectrum(energy = None, kv = 90, filtr = {'material':'Cu', 'density':8, 'thickness':0.1}, detector = {'material':'Si', 'density':5, 'thickness':1}):
    """
    Generate an effective specturm of a CT scanner.
    """            
    
    # Energy range:
    if not energy:
        energy = numpy.linspace(10, 90, 9)
        
    # Tube:
    spectrum = bremsstrahlung(energy, kv) 
    
    # Filter:
    if filtr:
        spectrum *= total_transmission(energy, filtr['material'], rho = filtr['density'], thickness = filtr['thickness'])
    
    # Detector:
    if detector:    
        spectrum *= scintillator_efficiency(energy, detector['material'], rho = detector['density'], thickness = detector['thickness'])
    
    # Normalize:
    spectrum /= (energy*spectrum).sum()
    
    return energy, spectrum
    
def spectralize(proj, kv = 90, n_phot = 1e8, specimen = {'material':'Al', 'density': 2.7}, filtr = {'material':'Cu', 'density':8, 'thickness':0.1}, detector = {'material':'Si', 'density':5, 'thickness':1}):
    """
    Simulate spectral data.
    """
    
    # Generate spectrum:
    energy, spectrum = effective_spectrum(kv, filtr, detector)
    
    # Get the material refraction index:
    mu = linear_attenuation(energy, specimen['material'], specimen['density'])
     
    # Display:
    display.plot(energy, spectrum, title = 'Spectrum') 
    display.plot(energy, mu, title = 'Linear attenuation') 
        
    # Simulate intensity images:
    counts = numpy.zeros_like(proj)
        
    for ii in range(len(energy)):
        
        # Monochromatic component:
        monochrome = spectrum[ii] * numpy.exp(-proj * mu[ii])
        monochrome = apply_noise(monochrome, 'poisson', n_phot) / n_phot    
        
        # Detector response is assumed to be proportional to E
        counts += energy[ii] * monochrome
    
    return counts

def forward_spectral(vol, proj, geometry, materials, energy, spectrum, n_phot = 1e8):
    """
    Simulate spectral data using labeled volume.
    """
    
    max_label = int(vol.max())
    
    if max_label != len(materials): raise ValueError('Number of materials is not the same as the number of labels in the volume!')

    # Normalize spectrum:
    spectrum /= (spectrum * energy).sum()
    
    # Simulate intensity images:
    lab_proj = []
    for jj in range(max_label):
        
        # Forward project:    
        proj_j = numpy.zeros_like(proj)
        vol_j = numpy.float32(vol == (jj+1))
        projector.forwardproject(proj_j, vol_j, geometry)
        
        lab_proj.append(proj_j)
        
    for ii in range(len(energy)):
        
        # Monochromatic components:
        monochrome = numpy.ones_like(proj)
        
        for jj in range(max_label):
            
            mu = linear_attenuation(energy, materials[jj]['material'], materials[jj]['density'])
    
            monochrome *= numpy.exp(-lab_proj[jj] * mu[ii])
            
        monochrome *= spectrum[ii]
        monochrome = apply_noise(monochrome, 'poisson', n_phot) / n_phot    

        # Detector response is assumed to be proportional to E
        proj += energy[ii] * monochrome
        
def material_refraction(energy, compound, rho):
    """    
    Calculate complex refrative index of the material taking
    into account it's density. 

    Args:
        compound (str): compound chemical formula
        rho (float): density in g / cm3
        energy (numpy.array): energy in KeV   

    Returns:
        float: refraction index in [1/mm]
    """
    cmp = xraylib.CompoundParser(compound)

    # Compute ration of Z and A:
    z = (numpy.array(cmp['Elements']))
    a = [xraylib.AtomicWeight(x) for x in cmp['Elements']]

    za = ((z / a) * numpy.array(cmp['massFractions'])).sum()

    # Electron density of the material:
    Na = phys_const['Na']
    rho_e = rho * za * Na

    # Attenuation:
    mu = mass_attenuation(energy, compound)

    # Phase:
    wavelength = 2 * numpy.pi * \
        (phys_const['h_bar_ev'] * phys_const['c']) / energy * 10

    # TODO: check this against phantoms.m:
    phi = rho_e * phys_const['re'] * wavelength

    # Refraction index (per mm)
    return rho * (mu / 2 - 1j * phi) / 10

def mass_attenuation(energy, compound):
    '''
    Total X-ray absorption for a given compound in cm2g. Energy is given in KeV
    '''   
    # xraylib might complain about types:
    energy = numpy.double(energy)

    if numpy.size(energy) == 1:
        return xraylib.CS_Total_CP(compound, energy)
    else:
        return numpy.array([xraylib.CS_Total_CP(compound, e) for e in energy])


def linear_attenuation(energy, compound, rho):
    '''
    Total X-ray absorption for a given compound in 1/mm. Energy is given in KeV
    '''
    # xraylib might complain about types:
    energy = numpy.double(energy)

    # unit: [1/mm]
    return rho * mass_attenuation(energy, compound) / 10


def compton(energy, compound):
    '''
    Compton scaterring crossection for a given compound in cm2g. Energy is given in KeV
    '''
    
    # xraylib might complain about types:
    energy = numpy.double(energy)
    import xraylib
    
    if numpy.size(energy) == 1:
        return xraylib.CS_Compt_CP(compound, energy)
    else:
        return numpy.array([xraylib.CS_Compt_CP(compound, e) for e in energy])


def rayleigh(energy, compound):
    '''
    Compton scaterring crossection for a given compound in cm2g. Energy is given in KeV
    '''
    import xraylib
    
    # xraylib might complain about types:
    energy = numpy.double(energy)

    if numpy.size(energy) == 1:
        return xraylib.CS_Rayl_CP(compound, energy)
    else:
        return numpy.array([xraylib.CS_Rayl_CP(compound, e) for e in energy])


def photoelectric(energy, compound):
    '''
    Photoelectric effect for a given compound in cm2g. Energy is given in KeV
    '''
    import xraylib
    
    # xraylib might complain about types:
    energy = numpy.double(energy)

    if numpy.size(energy) == 1:
        return xraylib.CS_Photo_CP(compound, energy)
    else:
        return numpy.array([xraylib.CS_Photo_CP(compound, e) for e in energy])


def scintillator_efficiency(energy, compound='BaFBr', rho=5, thickness=1):
    '''
    Generate QDE of a detector (scintillator). Units: KeV, g/cm3, mm.
    '''
    # Attenuation by the photoelectric effect:
    spectrum = 1 - numpy.exp(- thickness * rho *
                             photoelectric(energy, compound) / 10)

    # Normalize:
    return spectrum / spectrum.max()


def total_transmission(energy, compound, rho, thickness):
    '''
    Compute fraction of x-rays transmitted through the filter. 
    Units: KeV, g/cm3, mm.
    '''
    # Attenuation by the photoelectric effect:
    return numpy.exp(-linear_attenuation(energy, compound, rho) * thickness)


def bremsstrahlung(energy, energy_max):
    '''
    Simple bremstrahlung model (Kramer formula). Emax
    '''
    if energy_max < 10:
        
        warnings.warn('Maximum energy in the Kramer formula is too low. Setting to 100kV')
        energy_max = 100
        
    # Kramer:    
    spectrum = energy * (energy_max - energy)
    spectrum[spectrum < 0] = 0

    # Normalize:
    return spectrum / spectrum.max()

def gaussian_spectrum(energy, energy_mean, energy_sigma):
    '''
    Generates gaussian-like spectrum with given mean and STD.
    '''
    return numpy.exp(-(energy - energy_mean)**2 / (2 * energy_sigma**2))

def nist_names():
    '''
    Get a list of registered compound names understood by nist
    '''
    import xraylib
    
    return xraylib.GetCompoundDataNISTList()

def find_nist_name(compound_name):
    '''
    Get physical properties of one of the compounds registered in nist database
    '''
    import xraylib
    
    return xraylib.GetCompoundDataNISTByName(compound_name)

def parse_compound(compund):
    '''
    Parse chemical formula
    '''
    import xraylib
    
    return xraylib.CompoundParser(compund)