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
slpThresh1=(0,200)
slpThresh2=(0,200)
bpiThresh=1
minPeakSpacing=1


# slpfp = os.path.abspath("../../data/gscRasters/gsc_ps2_slope_AUVBath_utm.tif")
# bpifp = os.path.abspath("../../data/gscRasters/gsc_ps2_bpi_0311_AUVBath_utm.tif")
slpfp = os.path.abspath("../../../../rasterStorage/Endeavor/end_slope_AUVbathFilled.tif")
bpifp = os.path.abspath("../../../../rasterStorage/Endeavor/end_bpi_0311_AUVBathFilled.tif")


oname = "end_ps2_boolFilt_{}_{}_{}".format(ws,minPeakSpacing,bpiThresh).replace(".", "p")
onameRas = os.path.abspath("../../data/endRasters/{}.tif".format(oname))
onamePts = os.path.abspath("../../data/{}.csv".format(oname))

saveOut = False

r=(ws-1)//2
#%%

mainRaster, slopeTransform, sr, _ = readRaster(slpfp)
mainRaster = np.where(mainRaster < -10**37, np.nan, mainRaster)
mainRaster = mainRaster[0,...]
tempRaster = np.lib.stride_tricks.sliding_window_view(mainRaster, (ws,ws))

#%%
idx0 = np.flip(np.arange(r))
idx1 = np.arange(r)+1+r

spokeData = np.stack((tempRaster[:,:,idx0,r],
                      tempRaster[:,:,idx0,idx1],
                      tempRaster[:,:,r,idx1],
                      tempRaster[:,:,idx1,idx1],
                      tempRaster[:,:,idx1,r],
                      tempRaster[:,:,idx1,idx0],
                      tempRaster[:,:,r,idx0],
                      tempRaster[:,:,idx0,idx0]), axis=2)
del(tempRaster)

#%%
spokeCheck1 = ((spokeData >= slpThresh1[0]) & (spokeData <= slpThresh1[1]))
spokeCheck1 = np.any(spokeCheck1, axis=3)
spokeCheck1 = np.all(spokeCheck1, axis=2)

spokeCheck2 = ((spokeData >= slpThresh2[0]) & (spokeData <= slpThresh2[1]))
spokeCheck2 = np.any(spokeCheck2, axis=3)
spokeCheck2 = np.sum(spokeCheck2, axis=2)
spokeCheck2 = (spokeCheck2 > 3)

#%%

spokeCheck = (spokeData >= slpThresh1[0])
spokeCheck = np.any(spokeCheck, axis=3)
spokeCheck = np.all(spokeCheck, axis=2)

#%%
mainRaster, bpiTransform, sr, _ = readRaster(bpifp)
# mainRaster = np.where(mainRaster < -10**37, np.nan, mainRaster)
mainRaster = mainRaster[0,...]

#%%
# bpiCheck = (mainRaster>=bpiThresh)

bpiCheck = peak_local_max(mainRaster, min_distance=minPeakSpacing, indices=False, threshold_abs=bpiThresh)

# bpiCheck = findLocalMaxima(mainRaster, threshold=bpiThresh)
#%%
i0, j0 = getTransformIndex(0+r, 0+r, slopeTransform, bpiTransform)
i1, j1 = getTransformIndex(bpiCheck.shape[0]-r, bpiCheck.shape[1]-r, slopeTransform, bpiTransform)
  
# outputRaster =  np.all( np.stack((spokeCheck1, spokeCheck2, bpiCheck[i0:i1,j0:j1])), axis=0 )
outputRaster =  np.all( np.stack((spokeCheck, bpiCheck[i0:i1,j0:j1])), axis=0 )

print(np.sum(outputRaster))

#%%

outChimIndexes = np.argwhere(outputRaster)
outTransform = shiftTransform(slopeTransform, r, r)
outChimX, outChimY = getCoordsFromIndex(outChimIndexes[:,0],
                                        outChimIndexes[:,1],
                                        outTransform)


outChims = pd.DataFrame({"x_ll": outChimX, "y_ll": outChimY})

#%%
if(saveOut): 
    writeRaster(np.array(outputRaster, dtype=np.int8), outTransform, onameRas, sr)
    outChims.to_csv(onamePts)


#%%
###################################
scrEval.stop()
