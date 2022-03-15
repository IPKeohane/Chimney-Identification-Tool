# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 13:47:08 2021

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
import matplotlib.pyplot as plt
import glob

import rasterio
from rasterio.plot import show
from rasterio.transform import Affine
from rasterio.crs import CRS

import sklearn.feature_extraction.image as fe


#%%
fp = os.path.abspath("../../data/trainingData/citTrainPatches_gsc_2band_20220207/class0_ws15_b2_n20.tif")
fp = os.path.abspath("../../data/trainingData/citTrainPatches_gsc_2band_20220207/class1_ws15_b2_n21.tif")
fp = os.path.abspath("../../data/trainingData/citTrainPatches_gsc_2band_20220207/class2_ws15_b2_n20.tif")
fp = os.path.abspath("../../data/trainingData/citTrainPatches_gsc_2band_20220207/class3_ws15_b2_n20.tif")


testFiles = [os.path.abspath("../../data/trainingData/citTrainPatches_gsc_2band_20220207/class0_ws15_b2_n20.tif"),
             os.path.abspath("../../data/trainingData/citTrainPatches_gsc_2band_20220207/class1_ws15_b2_n21.tif"),
             os.path.abspath("../../data/trainingData/citTrainPatches_gsc_2band_20220207/class2_ws15_b2_n20.tif"),
             os.path.abspath("../../data/trainingData/citTrainPatches_gsc_2band_20220207/class3_ws15_b2_n20.tif")]

#%%

for fp in testFiles:

    
    #%%
    mainRaster, mainTransform, sr, _ = readRaster(fp)
    mainRaster = np.where(mainRaster < -10**37, np.nan, mainRaster)
    
    #%%
    plotBands(mainRaster)
    
    #%%
    print("b1 min: {}, b2 min: {}, b1 max: {}, b2 max: {}".format(
        np.nanmin(mainRaster[0,...]),
        np.nanmin(mainRaster[1,...]),
        np.nanmax(mainRaster[0,...]),
        np.nanmax(mainRaster[1,...])))


