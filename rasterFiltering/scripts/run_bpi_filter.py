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
from rasterFunctions import getTransformIndex, findLocalMaxima, breakUpRaster, stitchRasterChunks
from rasterFilters import directionalSlope, bathymetricPositionIndex
    
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
from pyproj import Proj
from skimage.feature import peak_local_max
from multiprocessing import Pool

import rasterio
from rasterio.plot import show
from rasterio.transform import Affine
from rasterio.crs import CRS

import sklearn.feature_extraction.image as fe

scrEval = startEval()
######


bthfp = os.path.abspath("../../../../rasterStorage/Endeavor/end_AUVBath_filled_1m.tif")
# bthfp = os.path.abspath("../../../../rasterStorage/GSC/gsc_AUVBath_filled_utm.tif")

ws_o = 11
ws_i = 3
r=(ws_o-1)//2

opath = "../../../../rasterStorage/Endeavor/end_bpi_{:02d}{:02d}_AUVBathFilled_1m.tif".format(ws_i, ws_o)
if(os.path.exists(opath)):
    opath = opath.replace(".tif", "_NEW.tif")
    
chunkShape = (4,1)

#%%
mainRaster, mainTransform, sr, _ = readRaster(bthfp)
mainRaster = np.where(mainRaster < -10**37, np.nan, mainRaster)
mainRaster = mainRaster[0,...]


#%%
# outRaster = bathymetricPositionIndex(mainRaster, ws_inner=ws_i, ws_outer=ws_o)

#%%
# transform = shiftTransform(mainTransform, r, r)

# writeRaster(outRaster, transform, 
#             oname=opath,
#             sr=sr,ftype='GTiff')


#%%
rasterChunks, chunkCoords = breakUpRaster(mainRaster, ws=ws_o, shape=chunkShape)

#%%
filterOutput = []
for m in range(chunkCoords.shape[0]):
        filterOutput.append([bathymetricPositionIndex(rasterChunks[m][n], ws_inner=ws_i, ws_outer=ws_o) for n in range(chunkCoords.shape[1])])

#%%
outRaster = stitchRasterChunks(filterOutput)

#%%
try:
    del rasterChunks
    del filterOutput
except:
    None

#%%
outRaster = np.pad(outRaster, r, constant_values=np.nan)

writeRaster(outRaster, mainTransform, 
            oname=opath,
            sr=sr,ftype='GTiff')

#%%
###################################
scrEval.stop()


