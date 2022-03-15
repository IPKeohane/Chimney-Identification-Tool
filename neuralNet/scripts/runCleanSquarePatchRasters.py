# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 10:47:11 2021

@author: Isaac
"""

import os, sys
module_path = os.path.abspath('../../src')
if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np

from rasterFunctions import cleanSquarePatchRasters
from citDataset import citDataset

#%%

trainingDataRoot = "../../data/trainingData/citTrainPatches_gsc_3band_20220214/"

cleanSquarePatchRasters(trainingDataRoot, verbose=10, deleteAnyNan=True)

#%%
dsFull = citDataset(path=trainingDataRoot)

#%%
dsFull.checkDataset()