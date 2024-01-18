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
from rasterFilters import directionalSlope
    
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
bpiThresh=1
minPeakSpacing=1


# bpifp = os.path.abspath("../../../../rasterStorage/GSC/gsc_ps2_bpi_0311_AUVBathFilled_utm.tif")
# bathfp = os.path.abspath("../../../../rasterStorage/GSC/gsc_ps2_AUVBath_filled_utm.tif")
bpifp  = os.path.abspath("../../../../rasterStorage/Endeavor/end_ps2_bpi_0311_AUVBathFilled.tif")
bathfp = os.path.abspath("../../../../rasterStorage/Endeavor/end_ps2_AUVBathFilled_medFilt.tif")


oname = "end_ps2_boolFilt_ws{}_sp{}_th{}".format(ws,minPeakSpacing,bpiThresh).replace(".", "p")
onameRas = os.path.abspath("../../data/endRasters/{}.tif".format(oname))
onamePts = os.path.abspath("../../data/{}.csv".format(oname))

saveOut = False

r=(ws-1)//2


#%%
bpiRaster,transform, sr, _ = readRaster(bpifp)
bpiRaster = np.where(bpiRaster < -10**37, np.nan, bpiRaster)
np.nan_to_num(bpiRaster, copy=False, nan=np.nan_to_num(-np.inf))

bpiRaster = bpiRaster[0,...]

#%%

bathRaster, _, _ , _ = readRaster(bathfp)
bathRaster = np.where(bathRaster < -10**37, np.nan, bathRaster)
np.nan_to_num(bathRaster, copy=False, nan=np.nan_to_num(-np.inf))
bathRaster = bathRaster[0,...]

#%%

bpiCheck = bpiRaster > bpiThresh

#%%

bthCheck = peak_local_max(bathRaster, min_distance=minPeakSpacing, indices=False)

#%%

outputRaster =  np.all( np.stack((bpiCheck, bthCheck)), axis=0 )

print(np.sum(outputRaster))

#%%

outChimIndexes = np.argwhere(outputRaster)
outChimX, outChimY = getCoordsFromIndex(outChimIndexes[:,0],
                                        outChimIndexes[:,1],
                                        transform)


outChims = pd.DataFrame({"x_ll": outChimX, "y_ll": outChimY})

#%%
if(saveOut): 
    writeRaster(np.array(outputRaster, dtype=np.int8), transform, onameRas, sr)
    outChims.to_csv(onamePts)


#%%
###################################
scrEval.stop()
