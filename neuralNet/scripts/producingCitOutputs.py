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
def produceCitOutputPoints(citOutputRaster, transform, min_distance=5, threshold=0.9, check3=False):
    
    localMaximaSci = peak_local_max(citOutputRaster[1,...],
                                     min_distance=min_distance,
                                     threshold_abs=0.5)

    class1check = citOutputRaster[1,localMaximaSci[:,0],localMaximaSci[:,1]]
    class3check = citOutputRaster[3,localMaximaSci[:,0],localMaximaSci[:,1]]
    
    sumcheck = ((class1check+class3check)>0.95)*check3
    threshcheck = class1check>threshold
    
    finalIndexes = localMaximaSci[np.logical_or(sumcheck, threshcheck),:]
    
    x1, y1 = getCoordsFromIndex(finalIndexes[:,0], finalIndexes[:,1], transform)
    
    outDf = pd.DataFrame({'x':x1, 'y':y1,
                          'class1':class1check[np.logical_or(sumcheck, threshcheck)],
                          'class3':class3check[np.logical_or(sumcheck, threshcheck)]})
    
    return outDf
    
    
    
    