#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 2017

@author: kostenko

Genereation of CT phantoms from geometrical primitives. Reads geometry data
to compute dimensions correctly.

"""
import numpy
import numpy.random
from scipy.ndimage import interpolation

def abstract_nudes(shape, geometry, complexity = 10):
    """
    Creates works of abstract art.
    """
    
    vol = random_spheroids(shape, geometry, overlap = 'xor', number = complexity)
    vol *= random_spheroids(shape, geometry, overlap = 'and', number = complexity)
    vol *= random_spheroids(shape, geometry, overlap = 'or', number = complexity)

    vol /= vol.max()
    
    return vol

def random_spheroids(shape, geometry, number = 3, overlap = 'xor', rotation = True):
    """
    Make a bunch of spheroids...
    """
    
    # Initialize volume:
    vol = numpy.zeros(shape, dtype = 'int')
    
    for ii in range(number):
        
        # Generate randomly:
        offset = _random_offset_(shape, geometry, 0.6)
        radii = numpy.abs(_random_size_(shape, geometry, 0.6))
        
        # Baby of a spheroid:
        sp = spheroid(shape, geometry, radii[0], radii[1], radii[2], offset = offset)
               
        # Rotate if needed:
        if rotation:
            sp = interpolation.rotate(sp, numpy.random.ranf() * 360, axes = (1,2), reshape=False)
        
        # Make bool:
        sp = sp > 0.5
        
        # Add to the hive:
        if overlap == 'or':
            vol = vol | sp
        elif overlap == 'and':
            vol = vol + sp
        elif overlap == 'xor':
            vol = vol + sp
            vol[vol > 1] = 0
        else:
            raise Exception('You Fool!')
            
    return vol.astype('float32')
                
def sphere(shape, geometry, r, offset = [0., 0., 0.]):
    """
    Make sphere. Radius is in units (geometry.parameters['unit'])
    """
    return spheroid(shape, geometry, r, r, r, offset)
    
def spheroid(shape, geometry, r1, r2, r3, offset = [0., 0., 0.]):
    """
    Make a spheroid. 
    """
    # Get the coordinates in mm:
    xx,yy,zz = geometry.volume_xyz(shape, offset)
    
    # Volume init: 
    return numpy.array((((xx[:, None, None]*r2*r3)**2 + (yy[None, :, None]*r1*r3)**2 + (zz[None, None, :]*r1*r2)**2) < (r1*r2*r3)**2), dtype = 'float32') 
    
def cuboid(shape, geometry, a, b, c, offset = [0., 0., 0.]):
    """
    Make a cuboid. Dimensions are in units (geometry.parameters['unit'])
    """
    # Get the coordinates in mm:
    xx,yy,zz = geometry.volume_xyz(shape, offset)
     
    return  numpy.array((abs(xx[:, None, None]) < a / 2) * (abs(yy[None, :, None]) < b / 2) * (abs(zz[None, None, :]) < c / 2), dtype = 'float32')  
    
       
def cylinder(shape, geometry, r, h, offset = [0., 0., 0.]):
    """
    Make a cylinder with a specified radius and height.
    """
    
    volume = numpy.zeros(shape, dtype = 'float32')
    
    # Get the coordinates in mm:
    xx,yy,zz = geometry.volume_xyz(shape, offset)
     
    volume = numpy.array(((zz[None, None, :])**2 + (yy[None, :, None])**2) < r ** 2, dtype = 'float32')  
    
    return (numpy.abs(xx) < h / 2)[:, None, None] * volume
        
def checkers(shape, geometry, frequency, offset = [0., 0., 0.]):
    """
    Make a 3D checkers board.
    """
    
    volume = numpy.zeros(shape, dtype = 'float32')
    
    # Get the coordinates in mm:
    xx,yy,zz = geometry.volume_xyz(shape, offset)
    
    volume_ = numpy.zeros(shape, dtype='bool')
    
    step = shape[1] // frequency
    
    for ii in range(0, frequency):
        sl = slice(ii*step, int((ii + 0.5) * step))
        volume_[sl, :, :] = ~volume_[sl, :, :]
    
    for ii in range(0, frequency):
        sl = slice(ii*step, int((ii + 0.5) * step))
        volume_[:, sl, :] = ~volume_[:, sl, :]

    for ii in range(0, frequency):
        sl = slice(ii*step, int((ii + 0.5) * step))
        volume_[:, :, sl] = ~volume_[:, :, sl]
 
    volume *= volume_
    
    return volume

def _random_offset_(shape, geometry, area_shrink = 1):
    """
    Generate random coordinates. Use area_shrink to shrink the area.
    """
    ranges = (numpy.array(shape) - 1) * geometry.voxel * area_shrink 
    return ranges * (numpy.random.rand(3) - 0.5)

def _random_size_(shape, geometry, area_shrink = 1):
    """
    Generate random sizes. It never produces zeros.
    """
    ranges = (numpy.array(shape) - 1) * geometry.voxel * area_shrink / 2
    res = ranges * (numpy.random.rand(3))
    res = (res * 4 / 5 + res.mean() / 5)
    
    return res