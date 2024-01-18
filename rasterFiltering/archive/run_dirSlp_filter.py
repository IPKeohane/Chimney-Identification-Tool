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

opath = "D:/Isaac/rasterStorage/Endeavor/end_dirSlpW_1m_utm.tif"
if(os.path.exists(opath)):
    opath = opath.replace(".tif", "_NEW.tif")
    
    
chunkShape = (6,1)
    

#%%
mainRaster, mainTransform, sr, _ = readRaster(bthfp)
mainRaster = np.where(mainRaster < -10**37, np.nan, mainRaster)
mainRaster = mainRaster[0,...]

print("original res: {}".format(mainRaster.shape))

#%%
rasterChunks, chunkCoords = breakUpRaster(mainRaster, ws=3, shape=chunkShape)

#%%
filterOutput = []
for c in range(chunkCoords.shape[0]):
    outChunk, _ = directionalSlope(rasterChunks[c][0], dx=mainTransform[0], dy=mainTransform[4])
    
    filterOutput.append([outChunk])

#%%
outRaster = stitchRasterChunks(filterOutput)
print("output res: {}".format(outRaster.shape))
#%%
try:
    del rasterChunks
    del mainRaster
    del filterOutput
    del outChunk
except:
    None

#%%
transform = shiftTransform(mainTransform, 1, 1)

writeRaster(outRaster, transform, 
            oname=opath,
            sr=sr,ftype='GTiff')

#%%
###################################
scrEval.stop()
