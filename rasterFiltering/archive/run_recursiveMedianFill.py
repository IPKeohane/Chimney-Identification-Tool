# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 22:53:44 2021

@author: Isaac
"""



import os, sys
module_path = os.path.abspath('../../src')
if module_path not in sys.path:
    sys.path.append(module_path)
    
from rasterFilters import recursizeMedianFill
from scriptEval import startEval
from rasterFunctions import writeRaster, plotBands, readRaster
    
import numpy as np

scrEval = startEval()
#%%
# fp = "../../../../rasterStorage/Endeavor/end_AUVBathMosaic_2.tif"
# fp = "../../../../rasterStorage/GSC/gsc_comp050059_AUVBath_utm.tif"
fp=

mainRaster, mainTransform, sr, mainMask = readRaster(fp)
mainRaster = np.where(mainRaster < -10e30,np.nan, mainRaster)


fp = "../../../../rasterStorage/Endeavor/end_mask_AUVBath.tif"
# fp = "../../../../rasterStorage/GSC/gsc_AUVBath_mask_utm.tif"
mask,  _, _, _ = readRaster(fp)

#%%

testRaster = np.where(mask==0, 999999, mainRaster)


#%%

testRasterOut = recursizeMedianFill(testRaster, verbose=10)


#%%

testRasterOut = np.where(testRasterOut>np.nanmax(mainRaster), np.nan, testRasterOut)


#%%

writeRaster(testRasterOut, mainTransform, "../../../../rasterStorage/Endeavor/end_AUVBath_filled.tif", sr)

scrEval.stop()







