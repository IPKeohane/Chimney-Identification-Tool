# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 13:04:00 2021

@author: Isaac
"""

import os, sys
module_path = os.path.abspath('../../src')
if module_path not in sys.path:
    sys.path.append(module_path)

from producingCitOutputs import produceCitOutputPoints, produceCitOutputRaster
from rasterFunctions import readRaster, writeRaster
from deepLearningFunctions import extractWindowsFromMask

from skimage.feature import peak_local_max

import numpy as np
from datetime import date


#%%

bandSet=[0,1,2]  # List of band indexes of the input raster to use. default=[0,1,2]
ws=15            # Patch window size (pixels). default=15
ncons=200        # Number of convolutional layers used in the model. default=200
nclasses=4       # Number of output classes. default=4

#%%
# Input raster file. Match this to the 3-band normalized raster derived from the 1m bathymetry
fp_in  = "../../data/cit_test_mulitbandRaster_gsc_1m.tif"  
# Filename for output point locations
opathDat = "../../data/default_citOutData.csv"
# Option to save the output raster in addition to point locations
saveRaster = False
# Filename for output raster.
opathRas = "../../data/default_citOutRaster.tif"
# path to the trained CNN model to use
modelPath = "../../neuralNet/models/citNN_20220214"
# Which output location method to use. Set True to extract all candidate locations
#  into memory before classifying (recommended for smaller rasters). Set False to
#  sample and classify each candidate location individually.
extract_not_sample = True

#%%
dt = date.today().strftime("%Y%m%d")
#%%

# load main raster
mainRaster, mainTransform, sr, _ = readRaster(fp_in)
mainRaster = np.where(mainRaster < -10e10,np.nan, mainRaster)
mainRaster = mainRaster[bandSet,...]

#%%
### calculate boolean raster identifying candidate locations to extract
bpiThresh = (1/20) # bpi threshold for candidate filter. Default=1/20
minPeakSpacing = 2 # minimum space between local maxima 
bpiCheck = mainRaster[2,...] > bpiThresh
bthCheck = np.zeros_like(mainRaster[0,...], dtype=bool)
local_indexes = peak_local_max(np.nan_to_num(mainRaster[0,...],copy=True,nan=0.0)
                               ,min_distance=minPeakSpacing)   # identifies local maxima 
bthCheck[local_indexes[:,0], local_indexes[:,1]] = True

# boolean raster indicating candidate locations to classify with the CNN
boolRaster =  np.all( np.stack((bpiCheck, bthCheck)), axis=0 )
boolTransform = mainTransform

#%%
# extract raster windows from candidate locations
windowTensors, locations = extractWindowsFromMask(boolRaster, mainRaster,
                                      boolTransform, mainTransform,
                                      ws=ws, asTorch = True, verbose=0)

#%%
# produce an output raster of model values 
citOutputRaster = produceCitOutputRaster(windowTensors, locations, mainRaster.shape,
                                          modelPath=modelPath,
                                          ws=ws, nbands=len(bandSet), ncons=ncons, nclasses=nclasses, verbose=0)

#%%
# identify point locations of CIT picks
citOutDf = produceCitOutputPoints(citOutputRaster, mainTransform,
                                  min_distance=3,
                                  threshold_u=0.95,
                                  threshold_l=0.3)
#%%
citOutDf['classOut_group'] = 2
citOutDf.loc[citOutDf['class2']>0.95, 'classOut_group'] = 1

#%%

citOutDf.to_csv(os.path.abspath(opathDat))

if(saveRaster):
    writeRaster(citOutputRaster[1,...], mainTransform, oname=opathRas,
                sr=sr,ftype='GTiff')


