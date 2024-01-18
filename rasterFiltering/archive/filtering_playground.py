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

import rasterio
from rasterio.plot import show
from rasterio.transform import Affine
from rasterio.crs import CRS

import sklearn.feature_extraction.image as fe

scrEval = startEval()
######


bthfp = os.path.abspath("D:/Isaac/rasterStorage/Endeavor/end_bth_1m_utm.tif")
opath = os.path.abspath("D:\Isaac\rasterStorage\Endeavor\end_bpi_0109_1m_utm.tif")

ws_o = 9
ws_i = 1
r=(ws_o-1)//2

#%%
mainRaster, mainTransform, sr, _ = readRaster(bthfp)
mainRaster = np.where(mainRaster < -10**37, np.nan, mainRaster)
mainRaster = mainRaster[0,...]

#%%
rasterChunks, chunkCoords = breakUpRaster(mainRaster, ws=ws_o, shape=(4,1))

#%%
filterOutput = []
for c in range(chunkCoords.shape[0]):
    filterOutput.append([bathymetricPositionIndex(rasterChunks[c][0], ws_inner=ws_i, ws_outer=ws_o)])

#%%
outRaster = stitchRasterChunks(filterOutput)

#%%
del rasterChunks
del mainRaster
del filterOutput

#%%
transform = shiftTransform(mainTransform, r, r)

writeRaster(outRaster, transform, oname=opath,
              sr=sr,ftype='GTiff')

#%%
###################################
scrEval.stop()
