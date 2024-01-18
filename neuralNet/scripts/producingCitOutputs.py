# -*- coding: utf-8 -*-
"""
Going from a boolean mask raster to a tiled array of local windows that can be
fed into a CNN classifier

Created on Mon Mar  1 14:53:12 2021

@author: Isaac Keohane -- isaackeohane95@gmail.com -- github: IPKeohane
"""

import os, sys
module_path = os.path.abspath('../../src')
if module_path not in sys.path:
    sys.path.append(module_path)
    
from rasterFunctions import getCoordsFromIndex
from chimNet import chimNet

from skimage.feature import peak_local_max
    
import numpy as np
import pandas as pd
import torch

#%%
def produceCitOutputRaster(windowTensors, locations, mainShape,
                           modelPath="../models/defaultModel", 
                           ws=15, nbands=3, ncons=150, nclasses=3, verbose=0):

    loaded_model = chimNet(ws, nbands, ncons, nclasses)
    loaded_model.load_state_dict(torch.load( os.path.abspath(modelPath) ))
    loaded_model.eval()

    outputRaster = np.array(np.zeros( (nclasses,mainShape[1],mainShape[2]) ),np.float64)
    
    for i in range(len(windowTensors)):
        
        if(ws!=windowTensors[i].shape[2]):
            r = windowTensors[i].shape[2]//2
            d = ws//2
            wtens = windowTensors[i][:,:,r-d:r+1+d,r-d:r+1+d]
        else:
            wtens = windowTensors[i]
        
        pred = loaded_model(wtens)
        pred = pred.detach().numpy()
        if(verbose>9): print(pred)
        outputRaster[:,locations[i,0],locations[i,1]] = pred[0,:]
    return outputRaster


#%%
def produceCitOutputPoints(citOutputRaster, transform, min_distance=5, threshold=0.9, checkSum=False):
    
    localMaximaSci = peak_local_max(citOutputRaster[1,...],
                                     min_distance=min_distance,
                                     threshold_abs=0.5)

    class1check = citOutputRaster[1,localMaximaSci[:,0],localMaximaSci[:,1]]
    class3check = citOutputRaster[3,localMaximaSci[:,0],localMaximaSci[:,1]]
    
    sumcheck = ((class1check+class3check)>0.95)*checkSum
    threshcheck = class1check>threshold
    
    finalIndexes = localMaximaSci[np.logical_or(sumcheck, threshcheck),:]
    
    x1, y1 = getCoordsFromIndex(finalIndexes[:,0], finalIndexes[:,1], transform)
    
    outDf = pd.DataFrame({'x':x1, 'y':y1,
                          'class1':class1check[np.logical_or(sumcheck, threshcheck)],
                          'class3':class3check[np.logical_or(sumcheck, threshcheck)]})
    
    return outDf
    
    
#%%
def produceCitOutputRasterSample(boolRaster, raster, boolTransform, transform,
                                       skip=10, modelPath="../models/defaultModel",
                                       ws=15, nbands=3, ncons=150, nclasses=3, normalize=False, verbose=0):
    
    #%%
    if(skip>0):
        boolRaster[::skip,::skip] = boolRaster[::skip,::skip] + 1
        boolRaster = np.where(boolRaster == 2, 1, 0)
    
    #%%
    loaded_model = chimNet(ws, nbands, ncons, nclasses)
    loaded_model.load_state_dict(torch.load( os.path.abspath(modelPath) ))
    loaded_model.eval()
    
    outputRaster = np.array(np.zeros((nclasses,
                                      np.shape(boolRaster[::skip,::skip])[0],
                                      np.shape(boolRaster[::skip,::skip])[1])), np.float32)
    #%%
    if(np.min(boolRaster)!=0 or np.max(boolRaster)!=1): # check that the boolean mask is in 0 and 1 
        print("ERROR: BOOLEAN MASK NOT BINARY FORMATTED")
        return
    if(len(raster.shape)==2):
        raster = np.expand_dims(raster,0)
        if(verbose>5): print("RESHAPED HxW RASTER TO BxHxW")
    indexes = np.argwhere(boolRaster) # get the indexes of the desired cells
    dc = (ws-1)//2
    imin, jmin, imax, jmax = dc, dc, raster.shape[1]-dc-1, raster.shape[2]-dc-1
    #%%
    for n in range(indexes.shape[0]):
    #%%
        i , j = getTransformIndex(indexes[n,0],indexes[n,1],boolTransform,transform)
        
        if(i<imin or j<jmin or i>imax or j>jmax):
            if(verbose>8): print("SKPIPPED {},{} - OUT OF BOUNDS".format(i,j))
            continue
        
        window = raster[:,i-dc:i+dc+1,j-dc:j+dc+1]
        
        if(np.isnan(window).any()):
            if(verbose>8): print("SKPIPPED {},{} - CONTAINS NAN".format(i,j))
            continue
       
        window = arrayToTensor(window,normalize=normalize)
        
        if(ws!=window.shape[2]):
            r = window.shape[2]//2
            d = ws//2
            window = window[:,:,r-d:r+1+d,r-d:r+1+d]
        
        pred = loaded_model(window)
        pred = pred.detach().numpy()
        #%%
        outputRaster[:,indexes[n,0]//skip,indexes[n,1]//skip] = pred[0,:]    
            
        if(verbose>7):
            if(n%(indexes.shape[0]//10)==0):
                print("Produce CIT output raster {}% done".format(round(n/indexes.shape[0]*100)))
    #%%            
    outTransform = Affine(boolTransform[0]*skip, boolTransform[1], boolTransform[2]-(skip/4),
                          boolTransform[3], boolTransform[4]*skip, boolTransform[5])    
        
    if(verbose>7): print("produceCitOutputRaster completed")
    return outputRaster, outTransform
     
    