#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 2018
@author: kostenko

This module some wrappers around ASTRA to make lives of people slightly less horrible.
A lone traveller seeking for some algorithms can find bits and pieces of SIRT, FISTA and more.
We will do our best to be memmap compatible and make sure that large data will not make your PC sink into dispair.
"""

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Imports >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import numpy                # arithmetics, arrays
import sys                  # error info
import traceback            # errors errors
import random               # random generator for blocking
import scipy                # minimizaer used in Students-T
from tqdm import tqdm       # progress bar
from time import sleep      # pause to allow time for the progress bar to print

import astra                       # The mother of tomography
import astra.experimental as asex  # The ugly offspring 

from flexdata import io     # geometry to astra conversions
from flexdata import array  
from flexdata import display# show images

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Global vars >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
          
# Here I will put some settings:

settings = {
'progress_bar' : True,      # Show progress bar. Now works only in FDK, backproject and forwardproject
'block_number' : 1,        # subsets or blocks into which the projections are divided
'mode' : 'sequential',      # This field can be 'random', 'sequential' or 'equidistant'
'poisson_weight' : False,   # use weights of projection pixels according to a Poisson statistics

'bounds' : None,            # bound reconstruction values
'preview' : False,          # show previews

'norm_update' : False,      # update norm during iterations?
'norm' : []}                 # stored norm

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def init_volume(projections, geometry = None):
    """
    Initialize a standard volume array.
    """          
    # Use geometry to compute additional offset if needed:
    
    if geometry:
        '''
        sample = geometry.det_sample

        offset = int(abs(geometry.parameters['vol_tra'][2]) / geometry.voxel / sample[2])
        '''
        shape = geometry.volume_shape(projections.shape)
        
    else:
        sample = [1, 1, 1]

        shape = projections[::sample[0], ::sample[1], ::sample[2]].shape
        shape = [shape[0], shape[2], shape[2]]
    
    return numpy.zeros(shape, dtype = 'float32')

def FDK( projections, volume, geometry):
    """
    Feldkamp, Davis and Kress cone beam reconstruction.
    Args:
        projections : input numpy.array (dtype = float32) with the following dimensions: [vrt, rot, hrz]
        volume      : output numpy.array (dtype = float32) with the following dimensions: [vrt, mag, hrz]
        geometry    : geometry description - one of threee types: 'simple', 'static_offsets', 'linear_offsets'
    """
    backproject(projections, volume, geometry, 'FDK_CUDA')

def backproject( projections, volume, geometry, algorithm = 'BP3D_CUDA', sign = 1):
    """
    Backproject using standard ASTRA functionality. If data array is memmap, backprojection is done in blocks to save RAM.
    Args:
        projections : input numpy.array (dtype = float32) with the following dimensions: [vrt, rot, hrz]
        volume      : output numpy.array (dtype = float32) with the following dimensions: [vrt, mag, hrz]
        geometry    : geometry description - one of threee types: 'simple', 'static_offsets', 'linear_offsets'
        algorithm   : ASTRA algorithm type ['BP3D_CUDA', 'FDK_CUDA' etc.]
        sign        : either +1 or -1 (add or subtract the data)
    """
    global settings
    block_number = settings['block_number']
        
    # Check if projections should be subsampled:
    sam = geometry.det_sample
    
    if sum(sam) > 3:
        projections = projections[::sam[0], ::sam[1], ::sam[2]]
        
    # Weight correction    
    prj_weight = _astra_norm_(projections, volume, geometry, algorithm)
    
    # If algorithm is FDK we use single-block projection unless data is a memmap
    if block_number == 1:
        
        projections = _contiguous_check_(projections) 
        
        # Initialize ASTRA geometries:
        vol_geom = geometry.astra_volume_geom(volume.shape)
        proj_geom = geometry.astra_projection_geom(projections.shape)  
        
        # Progress bar:
        #pbar = _pbar_start_(1)

        projections *= prj_weight
        
        # ASTRA here...
        _backproject_block_add_(projections, volume, proj_geom, vol_geom, algorithm, negative = (sign < 0))

        projections /= prj_weight
        
        #_pbar_update_(pbar)
        #_pbar_close_(pbar)
        
    else:   
        # Here is the multi-block version:
        
        # Initialize ASTRA volume geometry:
        vol_geom = geometry.astra_volume_geom(volume.shape)
        
        # Random mode may not work for FDK since it doesn't guarantee coverage for all angles...
        # Use mode = 'sequential'
        index = _block_index_(projections.shape[1])
        
        # Progress bar:
        pbar = _pbar_start_(block_number, 'block')
        
        # Loop over blocks:
        for ii in range(len(index)):
                
                proj_geom = geometry.astra_projection_geom(projections.shape, index[ii])  
                
                # number of projections in a block can vary a bit and FDK is not aware of that...
                block = projections[:, index[ii], :] * prj_weight * len(index[ii])
                
                # BP and FDK behave differently in terms of normalization:
                if (algorithm == 'BP3D_CUDA'):
                    block *= len(index)
                    
                block = _contiguous_check_(block)
    
                # Backproject:    
                _backproject_block_add_(block, volume, proj_geom, vol_geom, algorithm, negative = (sign < 0))  
                
                _pbar_update_(pbar)

            
        _pbar_close_(pbar)
        
        # ASTRA is not aware of the number of blocks:    
        volume /= projections.shape[1]
                       
def forwardproject( projections, volume, geometry, sign = 1):
    """
    Forwardproject using standard ASTRA functionality. If data array is memmap, projection is done in blocks to save RAM.
    Args:
        projections : output numpy.array (dtype = float32) with the following dimensions: [vrt, rot, hrz]
        volume      : input numpy.array (dtype = float32) with the following dimensions: [vrt, mag, hrz]
        geometry    : geometry description - one of threee types: 'simple', 'static_offsets', 'linear_offsets'
        sign        : either +1 or -1 (add or subtract the data)
    """
    global settings
    block_number = settings['block_number']
    
    # Check if projections should be subsampled:
    sam = geometry.vol_sample
    
    if sum(sam) > 3:
        volume = volume[sam[0], sam[1], sam[2]]
    
    # Non-memmap case is a single block:
    volume = _contiguous_check_(volume) 
        
    # Forward project will always use blocks:     
    if block_number == 1:
        
        # Initialize ASTRA geometries:
        vol_geom = geometry.astra_volume_geom(volume.shape)
        proj_geom = geometry.astra_projection_geom(projections.shape)  
        
        # Progress bar:
        #pbar = _pbar_start_(1)
        
        _forwardproject_block_add_(projections, volume, proj_geom, vol_geom, negative = (sign < 0))
        
        #_pbar_update_(pbar)
        #_pbar_close_(pbar)
        
    else:   
        # Multi-block:
        
        # Initialize ASTRA geometries:
        vol_geom = geometry.astra_volume_geom(volume.shape)
        
        # Progress bar:
        #pbar = _pbar_start_(unit = 'block', total=block_number)
        
        # Random mode may not work for forward projection since it doesn't guarantee coverage for all angles...
        # Use mode = 'sequential'.... it may not be true anymore...
        index = _block_index_(projections.shape[1])
        
        # Loop over blocks:
        for ii in range(len(index)):
            
            if len(index) > 0: 
        
                # Extract a block:
                proj_geom = geometry.astra_projection_geom(projections.shape, index[ii])    
                block = projections[:, index[ii],:]
                block = _contiguous_check_(block)
                
                # Backproject:    
                _forwardproject_block_add_(block, volume, proj_geom, vol_geom, negative = (sign < 0))  
                
                projections[:, index[ii],:] = block
                
                #_pbar_update_(pbar)
            
        #_pbar_close_(pbar)
   
def SIRT( projections, volume, geometry, iterations):
    """
    Simultaneous Iterative Reconstruction Technique.
    """     
    global settings
    preview = settings['preview']
    norm_update = settings['norm_update']
    bounds = settings['bounds']
                       
    print('Feeling SIRTy...')
    sleep(0.3)
    
    # Switch off progress bar if preview is on...
    if preview: 
        ncols = 0
    else:
        ncols = 50
        
    # Residual norms:    
    norms = []
        
    # Iterate:    
    for ii in tqdm(range(iterations),ncols=ncols):
        
        norm = L2_step(projections, volume, geometry)
            
        if norm_update:
            norms.append(norm)  
    
        # Apply bounds
        if bounds:
            numpy.clip(volume, a_min = bounds[0], a_max = bounds[1], out = volume)    
                
        # Preview
        if preview:
            display.slice(volume, dim = 1)
            
    if norm_update:   
         display.plot(norms, semilogy = True, title = 'Resudual L2')   
   
def EM( projections, volume, geometry, iterations):
    """
    Expectation Maximization
    """ 
    global settings
    preview = settings['preview']
    norm_update = settings['norm_update']
    
    # Make sure that the volume is positive:
    if volume.max() <= 0: 
        volume *= 0
        volume += 1
    elif volume.min() < 0: volume[volume < 0] = 0

    projections[projections < 0] = 0

    # Initialize L2:
    settings['norm'] = []
            
    print('Em Emm Emmmm...')
    sleep(0.5)
    
    # Switch off progress bar if preview is on...
    if preview: 
        ncols = 0
    else:
        ncols = 50
        
    # Go!    
    for ii in tqdm(range(iterations), ncols = ncols):
         
        # Update volume:
        EM_step(projections, volume, geometry)
                    
        # Preview
        if preview:
            display.slice(volume, dim = 1)
            
    if norm_update:   
         display.plot(settings['norm'], semilogy = True, title = 'Resudual norm')      

def FISTA( projections, volume, geometry, iterations, lmbda = 0):
    '''
    FISTA reconstruction. Right now there is no TV minimization substep here!
    '''
    global settings
    preview = settings['preview']
    norm_update = settings['norm_update']
    
    # Sampling:
    samp = geometry.det_sample
    
    #shp = numpy.array(projections.shape)
    #shp //= samp
                        
    # Initialize L2:
    norms = []
    t = 1
    
    volume_t = volume.copy()
    volume_old = volume.copy()
    
    # TV residual:
    sz = list(volume.shape)
    sz.insert(0, 3)
    volume_tv = numpy.zeros(sz, dtype = 'float32')

    print('FISTING in progress...')
    sleep(0.3)
        
    # Switch off progress bar if preview is on...
    if preview: 
        ncols = 0
    else:
        ncols = 50
        
    for ii in tqdm(range(iterations), ncols = ncols):
    
        # Subsample projections:
        if sum(samp) > 3:
            proj = projections[::samp[0], ::samp[1], ::samp[2]]
        else:
            proj = projections
                    
        # Fista!!!    
        norm = FISTA_step(proj, volume, volume_old, volume_t, volume_tv, t, geometry, lmbda = lmbda)
        norms.append(norm)
        
        # Preview
        if preview:
            display.slice(volume, dim = 1)
            
    if norm_update:   
         display.plot(norms, semilogy = True, title = 'Resudual norm')   
         
def CPLS(projections, volume, geometry, iterations, lambda_tv = 0.1):
    """
    Chambolle-Pock reconstruction with TV minimization.
    """    
    global settings
    preview = settings['preview']
    norm_update = settings['norm_update']
    
    # Sampling:
    samp = geometry.det_sample
    
    # Initialize L2:
    settings['norm'] = []
    
    # Init volumes:
    vol_bar = numpy.ascontiguousarray(numpy.zeros_like(volume))
    proj_p = numpy.ascontiguousarray(numpy.zeros_like(projections))
    vol_q = numpy.zeros(numpy.concatenate(((len([0,1,2]), ), volume.shape)), dtype='float32')

    # Switch off progress bar if preview is on...
    if preview: 
        ncols = 0
    else:
        ncols = 50
    
    print('Chambol & Pock are on it!')
    sleep(0.3)
        
    # Switch off progress bar if preview is on...
    if preview: 
        ncols = 0
    else:
        ncols = 50
        
    for ii in tqdm(range(iterations), ncols = ncols):
     # Update volume:
        if sum(samp) > 3:
            # Make step with subsampled data:
            CPLS_step(projections[::samp[0], ::samp[1], ::samp[2]], 
                      volume, vol_bar, vol_q, proj_p, geometry, lambda_tv)
        
        else:
            # Step with full data:
            CPLS_step(projections, volume, vol_bar, vol_q, proj_p, geometry, lambda_tv)
        
        # Preview
        if preview:
            display.slice(volume, dim = 1)
            
    if norm_update:   
         display.plot(settings['norm'], semilogy = True, title = 'Resudual norm')   
         
#def CPLS(projections, volume, geometry, iterations, lambda_tv):
    """
    Chambolle-Pock reconstruction with TV minimization.
    """ 
    #print('Work in progress....')
    """
    global settings
    preview = settings['preview']
    norm_update = settings['norm_update']
    
    # Sampling:
    samp = geometry.det_sample
    anisotropy = geometry.vol_sample
    
    shp = numpy.array(projections.shape)
    shp //= samp
    
    prj_weight = 1 / (shp[1] * numpy.prod(anisotropy) * max(volume.shape)) 
    
    # Initialize L2:
    settings['norm'] = []
    t = 1
    
    volume_t = volume.copy()
    volume_old = volume.copy()

    print('Chambol-Pock are doing their best...')
    sleep(0.3)
        
    # Switch off progress bar if preview is on...
    if preview: 
        ncols = 0
    else:
        ncols = 50
        
    for ii in tqdm(range(iterations), ncols = ncols):
     # Update volume:
        if sum(samp) > 3:
            proj = projections[::samp[0], ::samp[1], ::samp[2]]
            CPLS_step(proj, prj_weight, volume, volume_old, volume_t, t, geometry)
        
        else:
            CPLS_step(projections, volume, volume_old, volume_t, t, geometry)
        
        # Preview
        if preview:
            display.display_slice(volume, dim = 1)
            
    if norm_update:   
         display.plot(settings['norm'], semilogy = True, title = 'Resudual norm')   
         
         
         
        
    bounds = [0, 1000]    
    
    p = numpy.ascontiguousarray(numpy.zeros_like(proj))
    
    vol = project.init_volume(proj)
    vol_bar = project.init_volume(proj)
    
    sigma = numpy.ascontiguousarray(numpy.zeros_like(proj))
    project.forwardproject(sigma, (vol * 0 + 1), geom)
    sigma = sigma.max()
    
    #sigma = 1.0 / (sigma + (sigma == 0))
    #sigma_1 = 1.0  / (1.0 + sigma)
    
    tau = numpy.ascontiguousarray(numpy.zeros_like(vol))
    project.backproject(proj * 0 + 1, tau, geom)
    
    tau[(tau / numpy.max(tau)) < 1e-5] = numpy.inf
    tau = numpy.max(tau)
    #tau = tau # + norm of divergence
    #tau = 1 / tau
    #tau= sigma
    
    l2_norms = numpy.zeros(iterations)
    tru_norms = numpy.zeros(iterations)
    
    q = numpy.zeros(numpy.concatenate(((len([0,1,2]), ), vol.shape)), dtype='float32')
            
    # 
    for ii in tqdm.tqdm(range(iterations)):    
    
        res = proj.copy()
        project.forwardproject(res, -vol_bar, geom)
        
        p = (p + res * 1/sigma) / (1+1/sigma)
        
        #vol_new = vol.copy()
        vol_new = numpy.zeros_like(vol)
        project.backproject(p, vol_new, geom)
        
        if lambda_tv == 0:
            vol_new = vol + vol_new / tau
            
        else:
            q += numpy.stack(gradient(vol_bar)) / 2
            q /= numpy.fmax(lambda_tv, numpy.sqrt((q ** 2).sum(0)))
            q *= lambda_tv
            
            T = 1 / (tau + 2)
            vol_new = vol + vol_new * T + divergence(q) * T
            
        #vol_new *= tau
        #vol_new += vol
        
        numpy.clip(vol_new, a_min = bounds[0], a_max = bounds[1], out = vol_new)   
            
        vol_bar = vol_new + (vol_new - vol)
        vol = vol_new
        
        #display.display_slice(vol)
        
        l2_norms[ii] = scipy.linalg.norm(res)
        
        tru_norms[ii] = scipy.linalg.norm(vol -  vol0)
        
    display.plot(l2_norms[:], title = 'L2 norm')    
    display.plot(tru_norms[:], title = 'True norm')    
    
    display.display_slice(vol, title = 'vol')
    display.display_slice(vol0, title = 'true')
    
    print('True norm:', scipy.linalg.norm(vol -  vol0))
    """     
    
def MULTI_SIRT( projections, volume, geometries, iterations):
    """
    A multi-dataset version of SIRT. Here prjections and geometries are lists.
    """ 
    global settings
    preview = settings['preview']
    norm_update = settings['norm_update']
    norm = settings['norm']
    
    # Make sure array is contiguous (if not memmap):
    # if not isinstance(projections, numpy.memmap):
    #    projections = numpy.ascontiguousarray(projections)        
    
    # Initialize L2:
    norm = []

    print('Doing SIRT`y things...')
    sleep(0.5)

    # Switch off progress bar if preview is on...
    if preview: 
        ncols = 0
    else:
        ncols = 50
        
    for ii in tqdm(range(iterations), ncols = ncols):
        
        norm = 0
        for ii, proj in enumerate(projections):
            
            # This weight is half of the normal weight to make sure convergence is ok:
            #prj_weight = 1 / (proj.shape[1] * max(volume.shape)) 
    
            # Update volume:
            L2_step(proj, volume, geometries[ii])
                                    
        # Preview
        if preview:
            display.slice(volume, dim = 1)
            
    if norm_update:   
         display.plot(norm, semilogy = True, title = 'Resudual norm')        
         
def MULTI_PWLS( projections, volume, geometries, iterations = 10, student = False, weight_power = 1): 
    '''
    Penalized Weighted Least Squares based on multiple inputs.
    '''
    #error log:
    global settings
    norm = settings['norm']
    block_number = settings['block_number']
    mode = settings['mode']
    
    norm = []

    fac = volume.shape[2] * geometries[0].voxel * numpy.sqrt(2)

    print('PWLS-ing in progress...')
    sleep(0.5)
                              
    # Iterations:
    for ii in tqdm(range(iterations)):
    
        # Error:
        L_mean = 0
        
        # Create index slice to address projections:
        theta_n = projections[0].shape[1]
        index = _block_index_(theta_n)
        
        # Here we assume that theta_n is hte same for all projection datasets. TODO: fix this!!
            
        #Blocks:
        for jj in range(len(index)):        
            
            # Volume update:
            vol_tmp = numpy.zeros_like(volume)
            bwp_w = numpy.zeros_like(volume)
                       
            for kk, projs in enumerate(projections):
                            
                proj = numpy.ascontiguousarray(projs[:,index[jj],:])
                geom = geometries[kk]

                proj_geom = io.astra_proj_geom(geom, projs.shape, index = index[jj]) 
                vol_geom = io.astra_vol_geom(geom, volume.shape) 
            
                prj_tmp = numpy.zeros_like(proj)
                
                # Compute weights:
                if student:
                    fwp_w = numpy.ones_like(proj)
                    
                else:
                    me = proj.max() * weight_power / 5
                    fwp_w = numpy.exp(-proj * weight_power / me)
                                        
                #fwp_w = scipy.ndimage.morphology.grey_erosion(fwp_w, size=(3,1,3))
                
                _backproject_block_add_(fwp_w, bwp_w, proj_geom, vol_geom, 'BP3D_CUDA')
                _forwardproject_block_add_(prj_tmp, volume, proj_geom, vol_geom)
                
                prj_tmp = (proj - prj_tmp) * fwp_w / fac

                if student:
                    prj_tmp = _studentst_(prj_tmp, 5)
                    
                _backproject_block_add_(prj_tmp, vol_tmp, proj_geom, vol_geom, 'BP3D_CUDA')
                
                # Mean L for projection
                L_mean += (prj_tmp**2).mean() 
                
            eps = bwp_w.max() / 1000    
            bwp_w[bwp_w < eps] = eps
                
            volume += vol_tmp / bwp_w
            volume[volume < 0] = 0

            #print((volume<0).sum())
                
        norm.append(L_mean / len(index) / len(projections))
        
    display.plot(numpy.array(norm), semilogy=True)   
    
def L2_step(projections, volume, geometry):
    """
    A single L2 minimization step. Supports blocking and subsets.
    """
    global settings
    norm_update = settings['norm_update']
    poisson_weight = settings['poisson_weight']
        
    prj_weight = _astra_norm_(projections, volume, geometry, 'BP3D_CUDA')
      
    # Initialize ASTRA geometries:
    vol_geom = geometry.astra_volume_geom(volume.shape)      

    # Residual norm:    
    norm = 0
    
    # Split data into blocks:
    blocks, proj_geoms = _blocker_(projections, geometry)
    
    for ii, block in enumerate(blocks):
        
        # Extract a block:
        proj_geom = proj_geoms[ii]
                
        # Forwardproject:
        block_old = block.copy()
        _forwardproject_block_add_(block, volume, proj_geom, vol_geom, negative = True)   
                    
        # Take into account Poisson:
        if poisson_weight:
            # Some formula representing the effect of photon starvation...
            block *= numpy.exp(-block_old)
                
        # L2 norm (use the last block to update):
        if norm_update:
            norm = numpy.sqrt((block ** 2).mean())
        
        # Apply ramp to reduce boundary effects:
        #block = array.ramp(block, 0, 5, mode = 'linear')
        #block = array.ramp(block, 2, 5, mode = 'linear')
        
        block *= prj_weight * 2
        
        # Project
        _backproject_block_add_(block, volume, proj_geom, vol_geom, 'BP3D_CUDA')    
    
    return norm / len(blocks)
   
def CPLS_step(projections, vol, vol_bar, vol_q, proj_p, geometry, lambda_tv):         
    """
    A single CPLS minimization step
    """
    global settings
    norm_update = settings['norm_update']
    block_number = settings['block_number']
    bounds = settings['bounds']
    poisson_weight = settings['poisson_weight']
    mode = settings['mode']

    # Sampling:
    samp = geometry.det_sample
    
    shp = numpy.array(projections.shape)
    shp //= samp
    
    # Weight needed to make sure A and At are adjoint: 
    prj_weight = _astra_norm_(projections, vol, geometry, 'BP3D_CUDA')
    
    # Norm of A and At
    bp_weight = _backprojector_norm_(vol.shape, geometry)
    bp_weight = 0.39062488
    fp_weight = _forwardprojector_norm_(vol.shape, geometry)
    fp_weight = 3.5430822
    
    # Initialize ASTRA geometries:
    vol_geom = io.astra_vol_geom(geometry, vol.shape)      
    
    norm = 0
    
    for ii in range(block_number):
        
        # Create index slice to address projections:
        index = _block_index_(projections.shape[1])
        if len(index) == 0: break

        # Extract a block:
        proj_geom = geometry.astra_projection_geom(projections.shape, index = index)    
        
        # The block will contain the discrepancy eventually (that's why we need a copy):
        if (mode == 'sequential') & (block_number == 1):
            block = projections.copy()
            
        else:
            block = (projections[:, index, :]).copy()
            block = _contiguous_check_(block)
            
        # Forwardproject:
        _forwardproject_block_add_(block, vol_bar, proj_geom, vol_geom, negative = True)   
                    
        # Take into account Poisson:
        if poisson_weight:       
            # Some formula representing the effect of photon starvation...
            block *= numpy.exp(-projections[:, index, :])
            
        # Why prj_weight is not applyed after backprojection??    
        block *= block_number
        
        # Update p = p + (proj - vol_bar) / fp_weight / (1 + 1 / fp_weight)
        p_block = proj_p[:, index, :]
        p_block += block * 1/fp_weight
        p_block /= (1 + 1/fp_weight)
        
        proj_p[:, index, :] = p_block # proj_p doesnt seem to update otherwise...
        
        print(ii, '*************')
        print('p', p_block.max())
        
        # vol_new = vol.copy()
        # vol_new = At(p)
        vol_new = numpy.zeros_like(vol)
        _backproject_block_add_(p_block * prj_weight, vol_new, proj_geom, vol_geom, 'BP3D_CUDA')    
    
        print('bp', vol_new.max())
    
        display.slice(vol_new, title = 'vol_new')	
        
        if lambda_tv == 0:
            vol_new = vol + vol_new / bp_weight
            
        else:
            vol_q += numpy.stack(array.gradient(vol_bar)) / 2
            vol_q /= numpy.fmax(lambda_tv, numpy.sqrt((vol_q ** 2).sum(0)))
            vol_q *= lambda_tv
            
            vol_new = vol + (vol_new + array.divergence(vol_q)) / (bp_weight + 2)
         
        # Apply nonnegativity    
        if bounds:
            numpy.clip(vol_new, a_min = bounds[0], a_max = bounds[1], out = vol_new)   
            
        #vol_bar = vol_new + (vol_new - vol)
        vol_bar[:] = vol_new * 2
        vol_bar -= vol
        vol[:] = vol_new
        
        print('vol_bar', vol_bar.max())
        
        #display.slice(vol, title = 'vol')
        #display.slice(array.divergence(vol_q), title = 'div')
        print(scipy.linalg.norm(block))
                        
        # L2 norm (use the last block to update):
        if norm_update:
            norm = numpy.sqrt((block ** 2).mean())
          
    if norm_update:
        settings['norm'].append(norm / block_number)  

    # Apply bounds
    #if bounds:
    #    numpy.clip(vol, a_min = bounds[0], a_max = bounds[1], out = vol)         
        
def FISTA_step(projections, vol, vol_old, vol_t, vol_tv, t, geometry, lmbda = 0):
    """
    A single FISTA step. Supports blocking and subsets.
    """
    global settings
    bounds = settings['bounds']
    
    vol_old[:] = vol.copy()  
    vol[:] = vol_t.copy()
    
    t_old = t 
    t = (1 + numpy.sqrt(1 + 4 * t**2))/2
    
    L = 1 / _astra_norm_(projections, vol, geometry, 'BP3D_CUDA')
    
    # A*(A(x) - y):
    norm = L2_step(projections, vol_t, geometry)
    
    # Outside of the subsets loop:        
    if lmbda > 0:
        L1_step(vol_tv, vol_t, vol, L, lmbda)
    
    elif bounds is not None:
        vol[:] = numpy.clip(vol_t, a_min = bounds[0], a_max = bounds[1])  
        
    else:
        vol[:] = vol_t
    
    vol_t[:] = vol + ((t_old - 1) / t) * (vol - vol_old)
    
    return norm        
                
def L1_step(vol_tv, vol_t, vol, L, lamb):
    """
    Calculate image with lower TV. Stores the results in vol. It uses residual vol_tv from the last time it was called.
    """
    global settings
    bounds = settings['bounds']
    
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
        vol[:] = vol_t - la * array.divergence(final_vol_tv)
        
        if bounds is not None:
            numpy.clip(vol, a_min = bounds[0], a_max = bounds[1], out = vol)
                    
        # Taking a step towards minus of the gradient
        vol_tv_old = vol_tv.copy()
        vol_tv[:] = final_vol_tv - 1/(8*la) * array.gradient(vol)

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
    
def EM_step(projections, volume, geometry):
    """
    A single Expecrtation Maximization step. Supports blocking and subsets.
    """  
    global settings
    norm_update = settings['norm_update']
    block_number = settings['block_number']
    bounds = settings['bounds']
    mode = settings['mode']
    
    prj_weight = _astra_norm_(projections, volume, geometry, 'BP3D_CUDA') * 2
      
    # Initialize ASTRA geometries:
    vol_geom = geometry.astra_volume_geom(volume.shape)      
    
    # Norm update:
    norm = 0
    
    # Create index slice to address projections:
    index = _block_index_(projections.shape[1])
    
    for ii in range(len(index)):
        
        # Extract a block:
        proj_geom = geometry.astra_projection_geom(projections.shape, index = index[ii])    
        
        # Copy data to a block or simply pass a pointer to data itself if block is one.
        if (mode == 'sequential') & (block_number == 1):
            block = projections
            
        else:
            block = (projections[:, index[ii], :]).copy()
                    
        # Reserve memory for a forward projection (keep it separate):
        resid = _contiguous_check_(numpy.zeros_like(block))
        
        # Forwardproject:
        _forwardproject_block_add_(resid, volume, proj_geom, vol_geom)   
  
        # Compute residual:        
        resid[resid < resid.max() / 100] = numpy.inf  
        resid = (block / resid)
                    
        # L2 norm (use the last block to update):
        if norm_update:
            res_pos = resid[resid > 0]
            norm += res_pos.std() / res_pos.mean()
            
        # Project
        _backproject_block_mult_(resid * prj_weight * len(index), volume, proj_geom, vol_geom, 'BP3D_CUDA')    
    
    if norm_update:
        settings['norm'].append(norm / len(index))
    
    # Apply bounds
    if bounds:
        numpy.clip(volume, a_min = bounds[0], a_max = bounds[1], out = volume)               
        
def _pbar_start_(total, unit = 'it'):
    """
    If progress_bar is ON, initialize it.
    """        
    if settings['progress_bar']:
        return tqdm(total = total, unit = unit, ascii = True)
    
    else:
        return None
        
def _pbar_update_(pbar):
    if pbar:
        pbar.update()
        
def _pbar_close_(pbar):
    if pbar:
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
        
def _blocker_(projections, geometry):
    """
    Returns projection data split up in blocks.
    """
    # Create index slice to address projections:
    index = _block_index_(projections.shape[1])
    
    blocks = []
    proj_geoms = []
    
    # Subsample projections:
    samp = geometry.det_sample
    
    if sum(samp) > 3:
        proj = projections[::samp[0], ::samp[1], ::samp[2]]
    else:
        proj = projections
    
    # Divide in blocks:
    for ii in range(len(index)):
        
        block = (proj[:, index[ii], :]).copy()
        block = _contiguous_check_(block)
            
        blocks.append(block)
        proj_geoms.append(geometry.astra_projection_geom(proj.shape, index = index[ii]))
    
    return blocks, proj_geoms

def _astra_norm_(projections, volume, geometry, algorithm):
    """
    Compute a normalization factor in backprojection operator....
    """     
    # Check what is the real number of blocks:
    index = _block_index_(projections.shape[1])
    n = len(index)
    
    # This normalization is not done at ASTRA level at the moment:
    if algorithm == 'BP3D_CUDA':
        sam = geometry.det_sample
        anisotropy = geometry.vol_sample
        
        pix = (geometry.voxel[1]**4 * anisotropy[0] * anisotropy[1] * anisotropy[2] * anisotropy[2])
        return n / (projections.shape[1] // sam[1] * pix * max(volume.shape)) 
    
    else:
        return n       
        
def _backproject_block_add_(projections, volume, proj_geom, vol_geom, algorithm = 'BP3D_CUDA', negative = False):
    """
    Additive backprojection of a single block. 
    Use negative = True if you want subtraction instead of addition.
    """  
    
    _contiguous_check_(volume, allow_copy = False) 
         
    try:
        if negative:
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
        if algorithm == 'BP3D_CUDA':
            asex.accumulate_BP(projector_id, vol_id, sin_id)                
        elif algorithm == 'FDK_CUDA':
            asex.accumulate_FDK(projector_id, vol_id, sin_id)               
        else:
            raise ValueError('Unknown ASTRA algorithm type.')
            
        if isinstance(volume, numpy.memmap):
            volume[:] = vol_temp
            
                         
    except:
        # The idea here is that we try to delete data3d objects even if ASTRA crashed
        try:
            if negative:
                projections *= -1
        
            astra.algorithm.delete(projector_id)
            astra.data3d.delete(sin_id)
            astra.data3d.delete(vol_id)
            
        finally:
            info = sys.exc_info()
            traceback.print_exception(*info)        
    
    if negative:
        projections *= -1
    
    astra.algorithm.delete(projector_id)
    astra.data3d.delete(sin_id)
    astra.data3d.delete(vol_id)  
        
def _backproject_block_mult_( projections, volume, proj_geom, vol_geom, algorithm = 'BP3D_CUDA', operation = '+'):
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
        if algorithm == 'BP3D_CUDA':
            asex.accumulate_BP(projector_id, vol_id, sin_id)                
        elif algorithm == 'FDK_CUDA':
            asex.accumulate_FDK(projector_id, vol_id, sin_id)               
        else:
            raise ValueError('Unknown ASTRA algorithm type.')
                         
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
            
def _forwardproject_block_add_( projections, volume, proj_geom, vol_geom, negative = False):
    """
    Additive forwardprojection of a single block. 
    Use negative = True if you want subtraction instead of addition.
    """      

    _contiguous_check_(projections, allow_copy = False)      
    
    try:
        # We are goint to negate the projections block and not the whole volume:
        if negative:
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
        if negative:
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
    
def _contiguous_check_(data, allow_copy = True):
    '''
    Check if data is contiguous, if not - convert. This makes ASTRA happy.
    Careful, it may copy the data and overflow RAM.
    '''
    if not data.flags['C_CONTIGUOUS']:
        if not allow_copy:
            raise Exception('Data is not contiguous!')
            
        data = numpy.ascontiguousarray(data)
    
    # Check if data type is correct:
    if data.dtype != 'float32':
        if not allow_copy:
            raise Exception('Data type is not float32!')
            
        data = data.astype('float32')
        
    # Sometimes data shape is weird. Check.    
    if min(data.shape) == 0:
        raise Exception('Strange data shape:' + str(data.shape))
    
    if allow_copy:
        return data  

def _block_index_(proj_number):
    """
    Create index for projection blocks
    """   
    global settings
    block_number = settings['block_number']
    mode = settings['mode']
    
    # Length of the block and the global index:
    block_length = int(numpy.ceil(proj_number / block_number))
    index = numpy.arange(proj_number)

    # Different indexing modes:    
    if (mode == 'sequential')|(mode is None):
        # Index = 0, 1, 2, 3
        pass
        
    elif mode == 'random':   
        # Index = 2, 3, 0, 1 for instance...        
        random.shuffle(index)    
         
    elif mode == 'equidistant':   
        
        # Index = 0, 2, 1, 3   
        if proj_number > block_length:
            index = numpy.mod(numpy.arange(proj_number) * block_length, proj_number)
            
        else:
            # Equidistant formula doesnt work if block number == 1
            index = numpy.arange(proj_number)
        
    else:
        raise ValueError('Indexer type not recognized! Use: sequential/random/equidistant')
    
    index_out = []
    
    last = 0
    first = 0
    while last < proj_number:
        
        last = min(proj_number, first + block_length)
        
        if last > first:
            index_out.append(index[first:last])
            
        first = min(proj_number, first + block_length)    
    
    return index_out

def _studentst_( res, deg = 1, scl = None):
    '''
    StudentsT routine
    '''
    # nD to 1D:
    shape = res.shape
    res = res.ravel()
    
    # Optimize scale:
    if scl is None:    
        fun = lambda x: _misfit_(res[::70], x, deg)
        scl = scipy.optimize.fmin(fun, x0 = [1,], disp = 0)[0]
        #scl = numpy.percentile(numpy.abs(res), 90)
        #print(scl)
        #print('Scale in Student`s-T is:', scl)
        
    # Evaluate:    
    grad = numpy.reshape(_st_(res, scl, deg), shape)
    
    return grad

def _misfit_( res, scl, deg):
    
    c = -numpy.size(res) * (scipy.special.gammaln((deg + 1) / 2) - 
            scipy.special.gammaln(deg / 2) - .5 * numpy.log(numpy.pi*scl*deg))
    
    return c + .5 * (deg + 1) * sum(numpy.log(1 + numpy.conj(res) * res / (scl * deg)))
    
def _st_( res, scl, deg):   
    
    grad = numpy.float32(scl * (deg + 1) * res / (scl * deg + numpy.conj(res) * res))
    
    return grad
      

       
