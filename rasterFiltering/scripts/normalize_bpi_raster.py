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

files = ["G:/rasterStorage/Endeavor/end_bpi_0311_AUVBathFilled_1m.tif",
          "G:/rasterStorage/GSC/gsc_bpi_0311_AUVBath_filled_utm.tif"]

#%%
minvalue = -40
maxvalue = 40

#%%

for f in files:
    mainRaster, mainTransform, sr, _ = readRaster(f)
    mainRaster = np.where(mainRaster < -10**37,np.nan,mainRaster)

    # main_norm = exposure.rescale_intensity(mainRaster, in_range=(minvalue, maxvalue),out_range=(-1.0,1.0))
    main_norm = mainRaster/10

    writeRaster(main_norm.squeeze(), mainTransform,
                oname=f.replace(".tif", "_div10norm.tif"),
                sr=sr)



#%%
#################################
scrEval.stop()
#################################






