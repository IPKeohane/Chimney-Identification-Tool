# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 13:23:31 2021

@author: Isaac
"""


import os, sys
module_path = os.path.abspath('../../src')
if module_path not in sys.path:
    sys.path.append(module_path)
    
from scriptEval import startEval
#from rasterFunctions import shiftTransform, writeRaster, plotBands
    
import numpy as np

import rasterio


def directionalSlope(raster,dy=1.0,dx=1.0,verbose=0):
    '''
    Args:
        raster (2D numpy array): Input bathymetric raster as 2D numpy array.

    Returns:
        2 rasters of size m-2, n-2.  One with the North directional slope, the other with the East.
    '''
    if(verbose>0):
        print("Started direction slop calculation")
        scrEval = startEval()
    ws = 3
    dc = (ws-1)//2
    raster = np.pad(raster,dc)
    slpRasterN = (np.roll(raster,-1,0)-np.roll(raster,1,0))/((ws-1)*dy)
    slpRasterE = (np.roll(raster,-1,1)-np.roll(raster,1,1))/((ws-1)*dx)
    if(verbose>0):
        scrEval.stop()
        print("finished directional slope calculation")
    return slpRasterN[2*dc:(-2*dc),2*dc:(-2*dc)], slpRasterE[2*dc:(-2*dc),2*dc:(-2*dc)]
         



def booleanDirSlope(rasterN, rasterE, bpi, window=5,
                    bpiLimit=1.0, slpLimit=1.0, verbose=0):
    '''
    Moving window filter that returns an array where pixels are true if the surrounding
    window contains both +limit and -limit in both dir slope rasters
    
    Args:
        rasterN (TYPE): DESCRIPTION.
        rasterE (TYPE): DESCRIPTION.
        limit (TYPE, optional): DESCRIPTION. Defaults to 1.0.
        window (INT): size of the moving window to apply.
        verbose (TYPE, optional): DESCRIPTION. Defaults to 0.
       
    Returns:
        Array (int8): Returns a 2-d boolean array as an 8-bit integer array.
    '''    
    if(verbose>0):
        print("Started boolean dir slp filter calculation")
        scrEval = startEval()
    ws = window
    
    # boolean raster for E
    tempRaster = np.lib.stride_tricks.sliding_window_view(rasterE,(ws,ws))
    boolRasterE = (np.amin(tempRaster,axis=(2,3)) < -1*slpLimit) * (np.amax(tempRaster,axis=(2,3)) > slpLimit) 
    # boolean raster for N
    tempRaster = np.lib.stride_tricks.sliding_window_view(rasterN,(ws,ws))
    boolRasterN = (np.amin(tempRaster,axis=(2,3)) < -1*slpLimit) * (np.amax(tempRaster,axis=(2,3)) > slpLimit) 
    # boolean raster for bpi
    boolRasterBpi = bpi >= bpiLimit
    
    dc = (ws-1)//2
    
    if(verbose>0):
        scrEval.stop()
        print("finished boolean dir slp filter calculation")
    return np.array(boolRasterE * boolRasterN * boolRasterBpi[dc:-1*dc,dc:-1*dc], dtype=np.int8)
            

def bathymetricPositionIndex(raster, ws_inner=1, ws_outer=11):
    if(ws_inner%2==0 or ws_outer%2==0):
        print("ERROR: window sizes for BPI not odd numbers")
        return
    
    ri = (ws_inner-1)//2
    ro = (ws_outer-1)//2

    tempRaster = np.lib.stride_tricks.sliding_window_view(raster, (ws_outer, ws_outer))
    
    mask = np.array(np.ones(tempRaster.shape), dtype=bool)
    mask[:,:,ro-ri:ro+1+ri,ro-ri:ro+1+ri] = False
    
    outerMean = np.mean(tempRaster, axis=(2,3), where=mask)
    innerMean = np.mean(tempRaster, axis=(2,3), where=np.invert(mask))
    
    return innerMean-outerMean


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






              
    
    
    


