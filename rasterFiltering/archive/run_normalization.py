# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 16:33:21 2021

@author: Isaac Keohane -- isaackeohane95@gmail.com -- github: IPKeohane
"""

import os, sys
module_path = os.path.abspath('../../src')
if module_path not in sys.path:
    sys.path.append(module_path)
    
from scriptEval import startEval
from rasterFunctions import shiftTransform, writeRaster, plotBands, readRaster
    
import skimage
from skimage import exposure

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
####################################
scrEval = startEval()
####################################

#%%

files = ["../../../../rasterStorage/GSC/gsc_bpi_0311_AUVBath_filled_utm.tif",
         "../../../../rasterStorage/GSC/gsc_slope_AUVBathFilledMedFilt.tif",
         "../../../../rasterStorage/GSC/gsc_curv_AUVBathFilledMedFilt.tif",
         "../../../../rasterStorage/GSC/gsc_LoG151p8_AUVBathFilledBPI.tif"]


#%%

for f in files:
    mainRaster, mainTransform, sr, _ = readRaster(f)
    mainRaster = np.where(mainRaster < -10**37,np.nan,mainRaster)

    p1, p99 = np.nanpercentile(mainRaster, (0.1, 99.9))
    main_norm = exposure.rescale_intensity(mainRaster, in_range=(p1, p99),out_range=(-1.0,1.0))

    writeRaster(main_norm.squeeze(), mainTransform,
                oname=f.replace(".tif", "_norm.tif"),
                sr=sr)



#%%
#################################
scrEval.stop()
#################################






