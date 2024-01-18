#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:07:53 2021

@author: isaackeohane
"""

import os, sys
module_path = os.path.abspath('../../src')
if module_path not in sys.path:
    sys.path.append(module_path)
from scriptEval import startEval

from rasterFunctions import readRaster, writeRaster
from deepLearningFunctions import extractWindowsFromCoords
from citDataset import citDataset

import numpy as np
import pandas as pd
import torch
from datetime import date
from glob import glob
import matplotlib.pyplot as plt

#%%
ws=11

fp = "../../../../rasterStorage/Endeavor/end_ps2_5band_medFilt_filled_102321.tif"
mainRaster, mainTransform, sr, _ = readRaster(fp)
mainRaster = np.where(mainRaster < -10e30,np.nan, mainRaster)

fp = "../../data/end_chimLoc_ps2_combTrain_th095_3cpooled_20211023_synced.csv"
citPoints = pd.read_csv(fp)

trainingDataRoot = "../../data/trainingData/citTrainPatches_end_5band_medFilt_3cpooled_filled_102221/"
citDataset(path=trainingDataRoot, chunk=(ws,(0,1,2,3,4)))

#%%

pool = 1
#%%

coords = np.array( citPoints.loc[citPoints['gr_cat']==pool,['x','y']] )
#%%
ws_full=31
windowArrays, locations = extractWindowsFromCoords(coords, np.copy(mainRaster), mainTransform,
                                      ws=ws_full, asTorch = False, verbose=0)

#%%
bthLayers = []
for i in range(len(windowArrays)):
    norm = (windowArrays[i][0,:,:]-np.min(windowArrays[i][0,:,:]))/50.0
    bthLayers.append(norm)
    
bthWindows = np.stack(bthLayers)
windows = np.stack(windowArrays)

#%%
def radiusExtract(windows, r, b, d):
    l = 2*r+1
    c = np.concatenate((np.reshape(windows[:, b, d-r, d-r:d+r+1], (1,l,windows.shape[0])),
                        np.reshape(windows[:, b, d+r, d-r:d+r+1], (1,l,windows.shape[0])),
                        np.reshape(windows[:, b, d-r+1:d+r, d-r], (1,(l-2),windows.shape[0])),
                        np.reshape(windows[:, b, d-r+1:d+r, d+r].flatten(), (1,(l-2),windows.shape[0]))), axis=1 )
    return c
  

                  
#%%
b = 2
d = (windows.shape[2]-1)//2
radiusPts = [radiusExtract(windows, r, b, d) for r in range(d)]
radiusPts[0] = np.reshape(windows[:,b,d,d], (1,1,windows.shape[0]))


allvalues = np.zeros((0,2))

#%%
for r in range(len(radiusPts)):
    n = radiusPts[r].shape[2] * radiusPts[r].shape[1]
    brick = np.reshape(radiusPts[r], (n,1))
    
    allvalues = np.vstack( (allvalues, np.hstack((np.ones((n,1))*r, brick))) )


#%%
r=2
plt.hist(np.min(radiusPts[r], axis=(0,1)),  bins=30, range=(0,90))
plt.hist(np.max(radiusPts[r], axis=(0,1)),  bins=30, range=(0,90))

r=1
plt.hist(np.min(radiusPts[r], axis=(0,1)),  bins=30, range=(0,90))
plt.hist(np.max(radiusPts[r], axis=(0,1)),  bins=30, range=(0,90))





plt.hist(np.min(radiusPts[r][0], axis=1),  bins=30, range=(0,90))
plt.hist(np.max(radiusPts[r][0], axis=1),  bins=30, range=(0,90))

# r=1
# plt.hist(np.min(radiusPts[r][0], axis=1),  bins=30, range=(0,90))










