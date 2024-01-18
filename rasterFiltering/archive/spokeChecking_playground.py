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
from rasterFunctions import shiftTransform, writeRaster, plotBands, readRaster
from rasterFunctions import getTransformIndex, findLocalMaxima, getCoordsFromIndex
from deepLearningFunctions import extractWindowsFromCoords
from rasterFilters import directionalSlope
from citDataset import citDataset

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
from datetime import date
from pyproj import Proj
from skimage.feature import peak_local_max

import rasterio
from rasterio.plot import show
from rasterio.transform import Affine
from rasterio.crs import CRS

import sklearn.feature_extraction.image as fe

scrEval = startEval()
###########


#%%

ws=11
slpThresh1=(0,200)
slpThresh2=(0,200)


#%%
fp = "../../../../rasterStorage/Endeavor/end_ps2_5band_medFilt_filled_102321.tif"
mainRaster, mainTransform, sr, _ = readRaster(fp)
mainRaster = np.where(mainRaster < -10e30,np.nan, mainRaster)

fp = "../../data/end_chimLoc_ps2_combTrain_th095_3cpooled_20211023_synced.csv"
citPoints = pd.read_csv(fp)

#%%

trainingDataRoot = "../../data/trainingData/citTrainPatches_end_5band_medFilt_3cpooled_filled_102221/"
ds = citDataset(path=trainingDataRoot, chunk=(ws,(0,1,2,3,4)))
#%%
ws = int((100/15)*11)
trainingDataRoot = "../../data/trainingData/citTrainPatches_end_cont6band_medfilt_20211025/"
ds = citDataset(path=trainingDataRoot, chunk=(ws,(1,2,3,4,5)))
#%%
pool = 1

#%%

coords = np.array( citPoints.loc[citPoints['gr_cat']==pool,['x','y']] )

#%%
ws_full=31
windowArrays, locations = extractWindowsFromCoords(coords, np.copy(mainRaster), mainTransform,
                                      ws=ws_full, asTorch = False, verbose=0)

#%%
bthLayers_1 = []
slopeLayers_1 = []
for i in range(len(windowArrays)):
    norm = (windowArrays[i][0,:,:]-np.min(windowArrays[i][0,:,:]))/50.0
    bthLayers_1.append(norm)
    slopeLayers_1.append(windowArrays[i][2,:,:]*1.0)
    
bthWindows_1 = np.stack(bthLayers_1)
slpWindows_1 = np.stack(slopeLayers_1)
windows = np.stack(windowArrays)

#%%
bthLayers = []
slopeLayers = []
files = []
for i in range(len(ds)):
    if(ds[i][1]==1):
        bthLayers.append(ds[i][0][0,0,:,:].numpy())
        slopeLayers.append(ds[i][0][0,2,:,:].numpy())
        files.append(ds.getFile(i))


bthWindows = np.stack(bthLayers)
slpWindows = (np.stack(slopeLayers)+1.0) * (80.02/2)

#%%
r=(ws-1)//2

idx0 = np.flip(np.arange(r))
idx1 = np.arange(r)+1+r

spokeSlope = np.stack((slpWindows[:,idx0,r],
                       slpWindows[:,idx0,idx1],
                       slpWindows[:,r,idx1],
                       slpWindows[:,idx1,idx1],
                       slpWindows[:,idx1,r],
                       slpWindows[:,idx1,idx0],
                       slpWindows[:,r,idx0],
                       slpWindows[:,idx0,idx0]), axis=2)


slpMaxes = np.max( spokeSlope, axis=1  )
slpMinimumMaxes = np.min( slpMaxes, axis=1 )



#%%
ws=11
r=(ws-1)//2

idx0 = np.flip(np.arange(r))
idx1 = np.arange(r)+1+r

spokeSlope_1 = np.stack((slpWindows_1[:,idx0,r],
                       slpWindows_1[:,idx0,idx1],
                       slpWindows_1[:,r,idx1],
                       slpWindows_1[:,idx1,idx1],
                       slpWindows_1[:,idx1,r],
                       slpWindows_1[:,idx1,idx0],
                       slpWindows_1[:,r,idx0],
                       slpWindows_1[:,idx0,idx0]), axis=2)


slpMaxes_1 = np.max( spokeSlope_1, axis=1  )
slpMinimumMaxes_1 = np.min( slpMaxes_1, axis=1 )


















