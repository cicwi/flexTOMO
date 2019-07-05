#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 2018
@author: Alex Kostenko

This module contains: building blocks for reconstruction algorithms:
    back- and forward-projection operators, gradient descent updates 
     
All of the functions support large datasets implemented using numpy.memmap arrays
and geometry classes defined in flexData.geometry.

All projectors are based on ASTRA and are GPU-accelerated (CUDA).
"""

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Imports >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import numpy                # arithmetics, arrays
import sys                  # error info
import traceback            # errors errors
from scipy import optimize  # minimizer used in Students-T
from scipy import special  
from time import sleep      # Make a small pause to give time for the progress bar
from tqdm import tqdm       # progress bar

import astra                       # The mother of tomography
import astra.experimental as asex  # The ugly offspring 

from flexdata import display# show previews
from flexdata import data
from flexdata.data import logger

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Global settings >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class settings:
    '''
    Settings container used by projectors and reconstruction algorithms.
    Attributes:
        progress_bar    : show a progress bar     
        preview         : show previews
        update_residual : update the cost function
    
        subsets         : Number of projection subsets
        sorting         : Sorting of projections: 'sequential' or 'equidistant'
        
        poisson         : Weight pixels according to a Poisson statistics (only backprojection)
        student         : Use Students-T norm for regularization
        pixel_mask      : (scalar or 3d array) Mask applied to projections. If 3d, any dimension can be 0
        voxel_mask      : (scalar or 3d array) Mask applied to volume during forward projection. If 3d, any dimension can be 0
        fourier_filter  : (scalar or 2d array) Fourier filter applied to every projection (CTF)
        bounds          : Lower and upper bounds for the reconstruction values
        
    '''
    progress_bar = True      
    preview = False          
    update_residual = False  

    subsets = 1              
    sorting = 'sequential'   
    
    poisson = False   
    student = False       
    bounds = None              
    pixel_mask = None           
    voxel_mask = None        
    fourier_filter = None
   
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HIGH LEVEL ALGORITHMS >>>>>>>>>>>>>>>>>>>>
def init_volume(projections):
    """
    Initialize a standard-size volume array.
    """          
    sz = projections.shape
    return numpy.zeros((sz[0], sz[2], sz[2]), dtype = 'float32')    
    
def FDK( projections, volume, geometry):
    """
    Feldkamp, Davis and Kress cone beam reconstruction.
    Args:
        projections : input numpy.array (dtype = float32) with the following dimensions: [vrt, rot, hrz]
        volume      : output numpy.array (dtype = float32) with the following dimensions: [vrt, mag, hrz]
        geometry    : geometry description - one of threee types: 'simple', 'static_offsets', 'linear_offsets'
    """
    backproject(projections, volume, geometry, filtered = True)
   
def SIRT( projections, volume, geometry, iterations):
    """
    Simultaneous Iterative Reconstruction Technique.
    """ 
    ss = settings
    preview = ss.preview    
    bounds = ss.bounds
    
    logger.print('Feeling SIRTy...')
    
    # Residual norms:    
    rnorms = []
    
    # Progress bar:
    pbar = _pbar_start_(iterations, 'iterations')
        
    for ii in range(iterations):
        
        # L2 update:
        norm = l2_update(projections, volume, geometry)
        
        if norm:
            rnorms.append(norm)
        
        # Apply bounds
        if bounds:
            numpy.clip(volume, a_min = bounds[0], a_max = bounds[1], out = volume)    
       
        if preview:
            display.slice(volume, dim = 1, title = 'Preview')
        else:
            _pbar_update_(pbar)

    # Stop progress bar    
    _pbar_close_(pbar)
                     
    if rnorms:   
         display.plot2d(rnorms, semilogy = True, title = 'Resudual L2')   
         
def PWLS(projections, volume, geometry, iterations):
    """
    Simple implementation of the Penalized Weighted Least Squeares. 
    Gives better results when photon starvation and metal artifacts are present in small parts of the volume.
    Needs more memory than SIRT!
    """ 
    ss = settings
    preview = ss.preview    
    bounds = ss.bounds
    
    logger.print('PWLS-PWLS-PWLS-PWLS...')
    
    # Residual norms:    
    rnorms = []
    
    # Progress bar:
    pbar = _pbar_start_(iterations, 'iterations')
        
    for ii in range(iterations):
        
        # L2 update:
        norm = pwls_update(projections, volume, geometry)
        
        if norm:
            rnorms.append(norm)
        
        # Apply bounds
        if bounds:
            numpy.clip(volume, a_min = bounds[0], a_max = bounds[1], out = volume)    
       
        if preview:
            display.slice(volume, dim = 1, title = 'Preview')
        else:
            _pbar_update_(pbar)

    # Stop progress bar    
    _pbar_close_(pbar)
                     
    if rnorms:   
         display.plot(rnorms, semilogy = True, title = 'Resudual L2') 
         
def FISTA( projections, volume, geometry, iterations, lmbda = 0):
    '''
    FISTA reconstruction. Right now there is no TV minimization substep here!
    '''
    ss = settings
    preview = ss.preview    
    bounds = ss.bounds
        
    # Residual norms:    
    rnorms = []
    
    # Various variables:
    t = 1
    volume_t = volume.copy()
    volume_old = volume.copy()
    
    # TV residual:
    sz = list(volume.shape)
    sz.insert(0, 3)
    volume_tv = numpy.zeros(sz, dtype = 'float32')
    
    logger.print('FISTING started...')
    
    # Progress bar:
    pbar = _pbar_start_(iterations, 'iterations')
    
    for ii in range(iterations):
        
        # L2 update:
        norm = fista_update(projections, volume, volume_old, volume_t, volume_tv, t, geometry, lmbda = lmbda)
        
        if norm:
            rnorms.append(norm)
        
        # Apply bounds:
        if bounds:
            numpy.clip(volume, a_min = bounds[0], a_max = bounds[1], out = volume)    
         
        # Show preview or progress:    
        if preview:
            display.slice(volume, dim = 1, title = 'Preview')
        else:
            _pbar_update_(pbar)

    # Stop progress bar    
    _pbar_close_(pbar)
            
    if rnorms:   
         display.plot(rnorms, semilogy = True, title = 'Resudual norm')
   
def EM( projections, volume, geometry, iterations):
    """
    Expectation Maximization
    """ 
    ss = settings
    preview = ss.preview    
    bounds = ss.bounds
    
    # Make sure that the volume is positive:
    if (volume.max() == 0)|(volume.min()<0): 
        logger.error('Wrong initial guess. Make sure that initial guess for EM is positive.')
    
    if (projections.min() < 0):    
        logger.error('Wrong projection data. Make sure that projections have no negative values.')
    
    logger.print('Em Emm Emmmm...')
    
    # Residual norms:    
    rnorms = []
    
    # Progress bar:
    pbar = _pbar_start_(iterations, 'iterations')
        
    for ii in range(iterations):
        
        # Update volume:
        norm = em_update(projections, volume, geometry)
        
        # Apply bounds
        if bounds:
            numpy.clip(volume, a_min = bounds[0], a_max = bounds[1], out = volume) 
                               
        if norm:
            rnorms.append(norm)
            
        if preview:
            display.slice(volume, dim = 1, title = 'Preview')
        else:
            _pbar_update_(pbar)

    # Stop progress bar    
    _pbar_close_(pbar)
            
    if rnorms:   
         display.plot(rnorms, semilogy = True, title = 'Resudual norm') 
         
# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Low level building blocks>>>>>>>>>>>>>>>>>>>>>>    
def forwardproject(projections, volume, geometry, sign = 1):
    """
    Forwardproject using standard ASTRA functionality. 
    If projections array is numpy.memmap, projection is done in blocks to save RAM.
    
    Args:
        projections : output numpy.array (dtype = float32) with the following dimensions: [vrt, rot, hrz]
        volume      : input numpy.array (dtype = float32) with the following dimensions: [vrt, mag, hrz]
        geometry    : geometry description - one of threee types: 'simple', 'static_offsets', 'linear_offsets'
        sign        : either +1 or -1 (add or subtract the data)
    """
    subsets = settings.subsets
    
    # Non-memmap case is a single block:
    volume = _contiguous_check_(volume, copy = True) 

    # Progress bar:
    pbar = _pbar_start_(subsets, 'subsets')
     
    # Split data into subsets:
    for subset, pro_geom, vol_geom in _subset_generator_(projections, volume, geometry, copy = False):
        
        # Copy is made here to make sure the subset is contiguous:
        subset_c = _contiguous_check_(subset, copy = True) 
        
        _forwardproject_block_add_(subset_c, volume, pro_geom, vol_geom, sign)
        subset[:] = subset_c
        
        # Progress:
        _pbar_update_(pbar)
        
    # Stop progress bar    
    _pbar_close_(pbar)

def backproject(projections, volume, geometry, filtered = False, sign = 1):
    """
    Backproject using standard ASTRA functionality. 
    If data array is memmap, backprojection is done using 10+ subsets to save RAM.
    
    Args:
        projections : input numpy.array (dtype = float32) with the following dimensions: [vrt, rot, hrz]
        volume      : output numpy.array (dtype = float32) with the following dimensions: [vrt, mag, hrz]
        geometry    : geometry description. See flexData.geometry
        filtered    : use Feldkamp (True) or unfiltered (False) backprojection
        sign        : either +1 or -1 (add or subtract from volume)
    """
    # Get settings:
    subsets = settings.subsets
    
    # Weight correction    
    bp_weight = _bp_norm_(projections, volume, geometry)
    
    # Check if volume array is contiguous:
    volume = _contiguous_check_(volume, copy = False) 
    
    # Progress bar:
    pbar = _pbar_start_(subsets, 'subsets')
        
    for subset, pro_geom, vol_geom in _subset_generator_(projections, volume, geometry, copy = True):
        
        # Check projections:
        subset = _contiguous_check_(subset, copy = True) 
        
        # Normalize FDK and BP correctly:
        if filtered:
            subset *= subset.shape[1] / projections.shape[1]
        else:
            subset *= bp_weight 

        # Backproject:    
        _backproject_block_add_(subset, volume, pro_geom, vol_geom, filtered, sign)  

        # Progress:
        _pbar_update_(pbar)

    # Stop progress bar    
    _pbar_close_(pbar)
    
def l2_update(projections, volume, geometry):
    """
    A single L2-norm minimization update. Supports subsets.
    """
    # Global settings:
    update = settings.update_residual
    studentst = settings.student
    
    subsets = _subset_count_(projections)
        
    # Normalization factor:
    bp_weight = _bp_norm_(projections, volume, geometry)
    
    # Residual:    
    rnorm = 0
    
    # Split data into subsets:
    for subset, pro_geom, vol_geom in _subset_generator_(projections, volume, geometry, copy = True):
                
        # Forwardproject:
        residual = numpy.ascontiguousarray(numpy.zeros_like(subset))        
        _forwardproject_block_add_(residual, volume, pro_geom, vol_geom)   
            
        # Apply filters:
        residual = _filter_residual_(subset, residual)
        
        # L2 norm (use the last block to update):
        if update:
            rnorm += numpy.sqrt((residual**2).mean())
        
        # Apply StudentsT norm
        if studentst:
            degree = 6
            residual = _studentst_(residual, degree)
        
        # Project
        residual *= bp_weight * subsets * 2
        _backproject_block_add_(residual, volume, pro_geom, vol_geom, filtered  = False)    
        
    return rnorm / subsets

def pwls_update(projections, volume, geometry):
    """
    A single L2-norm update that applies weights based on Poisson 
    statistics to both residual and volume update. 
    Uses more memory than the standard l2_update.
    """
    # Global settings:
    update = settings.update_residual   
    subsets = _subset_count_(projections)
        
    # Normalization factor:
    bp_weight = _bp_norm_(projections, volume, geometry)
      
    # Residual:    
    rnorm = 0
    
    # Split data into subsets:
    for subset, pro_geom, vol_geom in _subset_generator_(projections, volume, geometry, copy = True):
                
        # Volume update:
        vol_tmp = numpy.zeros_like(volume)
        bwp_w = numpy.zeros_like(volume)
        
        # Compute weights:
        fwp_w = numpy.exp(-subset)
        _backproject_block_add_(fwp_w, bwp_w, pro_geom, vol_geom, filtered  = False)
        
        # Forwardproject:
        residual = numpy.ascontiguousarray(numpy.zeros_like(subset)) 
        _forwardproject_block_add_(residual, volume, pro_geom, vol_geom)   
        
        # Apply filters:
        residual = _filter_residual_(subset, residual) 
        
        # L2 norm (use the last block to update):
        if update:
            rnorm += numpy.sqrt((residual**2).mean())
        
        # Project
        residual *= bp_weight * subsets * 2 * fwp_w
        _backproject_block_add_(residual, vol_tmp, pro_geom, vol_geom, filtered  = False)    
            
        # Apply volume weights:
        bwp_w /= bwp_w.max()
        bwp_w[bwp_w < 1e-2] = 1e-2
                        
        volume += vol_tmp / bwp_w
        
    return rnorm / subsets

def fista_update(projections, vol, vol_old, vol_t, vol_tv, t, geometry, lmbda = 0):
    """
    A single FISTA step. Supports blocking and subsets.
    """
    bounds = settings.bounds
    subsets = _subset_count_(projections)
    
    vol_old[:] = vol.copy()  
    vol[:] = vol_t.copy()
    
    t_old = t 
    t = (1 + numpy.sqrt(1 + 4 * t**2))/2
    
    L = 1 / subsets
    
    # A*(A(x) - y):
    norm = l2_update(projections, vol_t, geometry)
    
    # Outside of the subsets loop:        
    if lmbda > 0:
        l1_update(vol_tv, vol_t, vol, L, lmbda)
    
    elif bounds is not None:
        vol[:] = numpy.clip(vol_t, a_min = bounds[0], a_max = bounds[1])  
        
    else:
        vol[:] = vol_t
    
    vol_t[:] = vol + ((t_old - 1) / t) * (vol - vol_old)
    
    return norm        
                
def l1_update(vol_tv, vol_t, vol, L, lamb):
    """
    Calculate image with lower TV. Stores the results in vol. It uses residual vol_tv from the last time it was called.
    """
    bounds = settings.bounds
    
    # Modified TV residual:
    final_vol_tv = vol_tv.copy()
    
    # these are some internal variables for this function:
    tau = 1
    stop_count = 0;
    ii = 0
    la = lamb / L

    # End result:
    vol[:] = vol_t * 0

    while ((ii < 6) & (stop_count < 3)):
        ii = ii + 1

        # old Z:
        vol_old = vol.copy();

        # new Xout:
        vol[:] = vol_t - la * data.divergence(final_vol_tv)
        
        if bounds is not None:
            numpy.clip(vol, a_min = bounds[0], a_max = bounds[1], out = vol)
                    
        # Taking a step towards minus of the gradient
        vol_tv_old = vol_tv.copy()
        vol_tv[:] = final_vol_tv - 1/(8*la) * data.gradient(vol)

        # this part can be changed to anisotropic, now it's L1 type:
        norm = numpy.zeros_like(vol)
    
        for jj in range(3):
            norm += vol_tv[jj]**2
        
        norm[norm < 1] = 1
        norm = numpy.sqrt(norm)    
        
        vol_tv /= norm[None, :]
        
                    
        #Updating residual and tau:
        tau_ = tau
        tau = (1 + numpy.sqrt(1 + 4*tau_**2)) / 2

        final_vol_tv = vol_tv + (tau_ - 1) / (tau) * (vol_tv - vol_tv_old)  

        # stop criterion:
        re = numpy.linalg.norm(vol - vol_old) / numpy.linalg.norm(vol);

        #print(re)
        
        if (re < 1e-3):
            stop_count = stop_count + 1;
        else:
            stop_count = 0;

def em_update(projections, volume, geometry):
    """
    A single Expecrtation Maximization step. Supports blocking and subsets.
    """  
    # Global settings:
    ss = settings
    update = ss.update_residual
    
    subsets = _subset_count_(projections)
        
    # Normalization factor:
    bp_norm = _bp_norm_(projections, volume, geometry)   

    # Residual:    
    rnorm = 0
    
    # Split data into subsets:
    for subset, pro_geom, vol_geom in _subset_generator_(projections, volume, geometry, copy = True):
        
        # Reserve memory for a forward projection (keep it separate):
        residual = numpy.zeros_like(subset)
        
        # Forwardproject:
        _forwardproject_block_add_(residual, volume, pro_geom, vol_geom)   

        # Compute residual:        
        residual[residual < residual.max() / 100] = numpy.inf  
        residual = (subset / residual)
                    
        # L2 norm (use the last block to update):
        if update:
            res_pos = residual[residual > 0]
            rnorm += res_pos.std() / res_pos.mean()
            
        # Project
        residual *= bp_norm * subsets * 2
        _backproject_block_mult_(residual, volume, pro_geom, vol_geom)    
       
    return rnorm / subsets
        
def _filter_residual_(projections, forward):    
    """
    Apply Fourier filter and Poisson weights to a residual
    """
    poisson = settings.poisson
    fourier_filter = settings.fourier_filter
    
    if fourier_filter is None:
            forward = projections - forward
            
    else:  
        # Apply Fourier filter:
        forward = fourier_filter * numpy.fft.fft2(projections, axes=(0, 2))
        forward -= (fourier_filter**2) * numpy.fft.fft2(forward, axes=(0, 2)) 
        forward = numpy.abs(numpy.fft.ifft2(forward, axes=(0, 2))).astype('float32')            

    # Apply Poisson weights with some scaling:
    if poisson:
        forward *= numpy.exp(-projections / projections.mean())
    
    return forward
        
def _pbar_start_(total, unit = 'it', ascii = False):
    """
    If progress_bar is ON, initialize it.
    """  
    sleep(0.3)
       
    if settings.progress_bar:
        return tqdm(total = total, unit = unit, ascii = ascii)
    
    else:
        return None
        
def _pbar_update_(pbar):
    if pbar:
        pbar.update()
        
def _pbar_close_(pbar):
    
    if pbar:    
        if pbar.total > pbar.n:
            pbar.update(pbar.total-pbar.n)        
        pbar.close()

def _forwardprojector_norm_(vol_shape, geometry):
    """
    Norm of the forward projection. Obtained through reverse engeneering.
    """
    # We assume that the longest possible ray equal to the diagonal of the volume:
    width = numpy.sqrt((numpy.array(vol_shape)**2).sum())
    
    return width * geometry.voxel

def _backprojector_norm_(vol_shape, geometry):
    """
    Norm of the forward projection. Obtained through reverse engeneering.
    """
    # We assume that the longest possible ray equal to the diagonal of the volume:
    width = numpy.sqrt((numpy.array(vol_shape)**2).sum())
    
    return 1 / (geometry.voxel * width)
        
def _bp_norm_(projections, volume, geometry):
    """
    Compute a normalization factor in backprojection operator....
    """       
    if type(geometry) == list:
        # This is a dirty fix, assuming that if a list of geometries is provided, they have same voxel and pixel sizes.
        vv = geometry[0].voxel
        det_sam = geometry[0].det_sample[1]
        
        pix = (vv[1] * vv[1] * vv[2] * vv[2]) / det_sam
        return 1 / (projections[0].shape[1] * pix * max(volume.shape))
    
    else:
        vv = geometry.voxel
        det_sam = geometry.det_sample[1]
    
        pix = (vv[1] * vv[1] * vv[2] * vv[2]) / det_sam
        return 1 / (projections.shape[1] * pix * max(volume.shape))
        
def _backproject_block_add_(projections, volume, proj_geom, vol_geom, filtered = False, sign = 1):
    """
    Additive backprojection of a single block. 
    Use negative = True if you want subtraction instead of addition.
    """  
    _contiguous_check_(volume, copy = False) 
         
    try:
        if sign < 0:
            projections *= -1
            
        # If volume is a memmap - create a temporary copy in RAM    
        if isinstance(volume, numpy.memmap):
            vol_temp = numpy.ascontiguousarray(volume)
            
            vol_id = astra.data3d.link('-vol', vol_geom, vol_temp)    
        else:
            vol_id = astra.data3d.link('-vol', vol_geom, volume)    
                    
        sin_id = astra.data3d.link('-sino', proj_geom, projections)        
        
        
        projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
        
        # We are using accumulate version to avoid creating additional copies of data.
        if not filtered:
            asex.accumulate_BP(projector_id, vol_id, sin_id)                
        else:
            asex.accumulate_FDK(projector_id, vol_id, sin_id)               
            
        if isinstance(volume, numpy.memmap):
            volume[:] = vol_temp
                         
    except:
        # The idea here is that we try to delete data3d objects even if ASTRA crashed
        try:
            if sign < 0:
                projections *= -1
        
            astra.algorithm.delete(projector_id)
            astra.data3d.delete(sin_id)
            astra.data3d.delete(vol_id)
            
        finally:
            info = sys.exc_info()
            traceback.print_exception(*info)        
    
    if sign < 0:
        projections *= -1
    
    astra.algorithm.delete(projector_id)
    astra.data3d.delete(sin_id)
    astra.data3d.delete(vol_id)  
        
def _backproject_block_mult_( projections, volume, proj_geom, vol_geom):
    """
    Multiplicative backprojection of a single block. 
    """           
    try:
        # Need to create a copy of the volume:
        volume_ = numpy.zeros_like(volume)
        
        sin_id = astra.data3d.link('-sino', proj_geom, projections)        
        vol_id = astra.data3d.link('-vol', vol_geom, volume_)    
        
        projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
        
        # We are using accumulate version to avoid creating additional copies of data.
        asex.accumulate_BP(projector_id, vol_id, sin_id)                
                         
    except:
        # The idea here is that we try to delete data3d objects even if ASTRA crashed
        try:
           
            astra.algorithm.delete(projector_id)
            astra.data3d.delete(sin_id)
            astra.data3d.delete(vol_id)
            
        finally:
            info = sys.exc_info()
            traceback.print_exception(*info)        

    volume *= volume_
    
    astra.algorithm.delete(projector_id)
    astra.data3d.delete(sin_id)
    astra.data3d.delete(vol_id)      
            
def _forwardproject_block_add_( projections, volume, proj_geom, vol_geom, sign =1 ):
    """
    Additive forwardprojection of a single block. 
    Use negative = True if you want subtraction instead of addition.
    """      
    try:
        # We are goint to negate the projections block and not the whole volume:
        if sign < 0:
            projections *= -1
            
        # If volume is a memmap - create a temporary copy in RAM    
        if isinstance(volume, numpy.memmap):
            vol_temp = numpy.ascontiguousarray(volume)
            
            vol_id = astra.data3d.link('-vol', vol_geom, vol_temp)    
        else:
            vol_id = astra.data3d.link('-vol', vol_geom, volume)        
                
        sin_id = astra.data3d.link('-sino', proj_geom, projections)        
        
        projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
        
        # Project!
        asex.accumulate_FP(projector_id, vol_id, sin_id)
        
        # Negate second time:
        if sign < 0:
            projections *= -1
             
    except:
        # Always try to delete data3d:
        try:
            astra.algorithm.delete(projector_id)
            astra.data3d.delete(sin_id)
            astra.data3d.delete(vol_id)   
        finally:
            info = sys.exc_info()
            traceback.print_exception(*info)
        
    astra.algorithm.delete(projector_id)
    astra.data3d.delete(sin_id)
    astra.data3d.delete(vol_id)   
    
def _contiguous_check_(data, copy = True):
    '''
    Check if data is contiguous, if not - convert. This makes ASTRA happy.
    Careful, it may copy the data and overflow RAM.
    '''
    if not data.flags['C_CONTIGUOUS']:
        if not copy:
            raise Exception('Data is not contiguous!')
        
        else:    
            data = numpy.ascontiguousarray(data)
    
    # Check if data type is correct:
    if data.dtype != 'float32':
        if not copy:
            raise Exception('Data type is not float32!')
        
        else:
            data = data.astype('float32')
        
    # Sometimes data shape is weird. Check.    
    if min(data.shape) == 0:
        raise Exception('Strange data shape:' + str(data.shape))
    
    return data  

def _studentst_(res, deg = 1, scl = None):
    '''
    StudentsT routine
    '''
    # nD to 1D:
    shape = res.shape
    res = res.ravel()
    
    # Optimize scale:
    if scl is None:    
        fun = lambda x: _misfit_(res[::100], x, deg)
        scl = optimize.fmin(fun, x0 = [1,], disp = 0)[0]
        #scl = numpy.percentile(numpy.abs(res), 90)
        #print(scl)
        #print('Scale in Student`s-T is:', scl)
        
    # Evaluate:    
    grad = numpy.reshape(_st_(res, scl, deg), shape)
    
    return grad

def _misfit_( res, scl, deg):
    
    if scl <= 0: return numpy.inf
    
    c = -numpy.size(res) * (special.gammaln((deg + 1) / 2) - 
            special.gammaln(deg / 2) - .5 * numpy.log(numpy.pi*scl*deg))
    
    return c + .5 * (deg + 1) * sum(numpy.log(1 + numpy.conj(res) * res / (scl * deg)))
    
def _st_( res, scl, deg):   
    
    grad = numpy.float32(scl * (deg + 1) * res / (scl * deg + numpy.conj(res) * res))
    
    return grad
      
def _subset_count_(projections):
    """
    Count how many actual subsets we have, taking into account indexing type and total data size.
    """
    count = 0
    if type(projections) != list:
        projections = [projections,]
        
    for sl in _slice_generator_(projections[0]):
        count += 1
        
    return count
    
def _slice_generator_(projections):
    """
    Generator of data indexing for subsets. Supports sequiential and equidistant indexing.
    """   
    ss = settings
    subsets = ss.subsets
    sorting = ss.sorting
    
    proj_n = projections.shape[1]
    
    if (sorting == 'sequential'):
    
        # Length of the block and the global index:
        step = int(numpy.ceil(proj_n / subsets))
    
        last, first = 0, 0
        while last < proj_n:
            
            last = min(proj_n, first + step)
            
            yield slice(first, last) 
            first = last
    
    elif (sorting == 'equidistant'):
        
        first = 0
        while first < subsets:
            
            yield slice(first, None, subsets) 
            first += 1
    else:
        logger.error('Unknown sorting!')
                   
def _subset_generator_(projections, volume, geometry, copy = False):
    """
    Generator of subsets for back-projection. Projections may be a single numpy array or a list of arrays.
    """
    ss = settings
    mask = ss.pixel_mask
    
    # If projections are not list (single dataset) - make it a list:
    if type(projections) != list:
        projections = [projections,]
    
    if type(geometry) != list:
        geometry = [geometry,]
    
    # Divide in subsets:   
    for sl in _slice_generator_(projections[0]):
        for jj, dataset in enumerate(projections):    
            
            subset = dataset[:, sl, :]
            if copy:
                subset = subset.copy()
            
            # Apply mask:
            if mask is not None:
                if mask.shape[1] == 1: 
                    # If mask is constant
                    subset *= mask
                else:
                    # If mask is defined per angle:
                    subset *= mask[:, sl, :]
            
            # Initialize ASTRA geometries:
            proj_geom = geometry[jj].astra_projection_geom(projections[jj].shape, index = sl)
            vol_geom = geometry[jj].astra_volume_geom(volume.shape)
     
            yield subset, proj_geom, vol_geom 