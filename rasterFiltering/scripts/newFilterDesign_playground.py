# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 21:23:46 2021

@author: Isaac
"""

import os, sys
module_path = os.path.abspath('../../src')
if module_path not in sys.path:
    sys.path.append(module_path)
    
from scriptEval import startEval
#from rasterFunctions import shiftTransform, writeRaster, plotBands
    
import numpy as np
import scipy

import rasterio

def recursizeMedianFill(raster, nanvalue = None, verbose=0):
    '''
    Parameters
    ----------
    raster (NUMPY ARRAY): Input raster as numpy array.
        dimensions: either height x width or bands x height x width
    nanvalue (VALUE): (optional) if the input raster doesn't have actual nan values
        in it, specify what the nan value is.
    verbose (INTEGER): If verbose > 5, will print the number of iterations and
        the number of holidays filled. If verbose > 8 it will print after each 
        iteration.

    Returns
    -------
    raster (NUMPY ARRAY): The input raster after repeated median filtering and 
        filling of holidays is performed until all NaNs are removed.
        dimensions: either height x width or bands x height x width
    '''
    
    if(nanvalue):
        raster = np.where(raster == nanvalue,np.nan, raster)
    if(np.sum(np.isnan(raster))==0):
        return raster
    if(len(raster.shape)==2):
        singleBand = True
        raster = np.expand_dims(raster,0)
    else:
        singleBand = False
    if(len(raster.shape)!=3):
        print("ERROR INCORRECT RASTER DIMENSION IN RECURSIZE MEDIAN FILL")
        return
    for b in range(raster.shape[0]):
        grid = raster[b,:,:]
        totalnans = np.sum(np.isnan(grid))
        hasnan = True
        i = 1
        while hasnan:
            Nnans = np.sum(np.isnan(grid))
            medFilt = np.nanmedian(np.lib.stride_tricks.sliding_window_view(
                np.pad(grid,1,'edge'), (3,3)), axis=(2,3))
            grid = np.where(np.isnan(grid),
                            medFilt,
                            grid)
            if(np.sum(np.isnan(grid))==0):
                hasnan=False
            if(verbose>8):
                print("Band {}, iteration {}: {}/{} holidays filled".format(
                    b, i, (Nnans-np.sum(np.isnan(grid))), Nnans ))
            i+=1
        if(verbose>5):
            print("Band {}: total iterations = {} | holidays filled = {}".format(
                    b, i-1, totalnans ))
        raster[b,:,:] = grid
    if(singleBand):
        raster = raster[0,:,:]
    return raster

        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    