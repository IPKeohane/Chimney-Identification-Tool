# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 13:04:00 2021

@author: Isaac
"""

import os, sys
module_path = os.path.abspath('../../src')
if module_path not in sys.path:
    sys.path.append(module_path)
from scriptEval import startEval

from producingCitOutputs import produceCitOutputPoints, produceCitOutputRaster
from rasterFunctions import readRaster, writeRaster
from deepLearningFunctions import extractWindowsFromMask

from chimNet import chimNet
from citDataset import citDataset

import numpy as np
import pandas as pd
import torch
from datetime import date
from glob import glob


#%%

bandSet=[0,1,2]
ws=15
ncons=200
nclasses=4


#%%

# modelPath = "../../neuralNet/models/citNN_comb_3cpooled_medFilt/citNN_comb_20211023_ws15_nc200_bs01234_ne8"
# modelPath = "../../neuralNet/models/citNN_comb_3cpooled_medFilt/citNN_comb_20211023_ws15_nc200_bs01234_ne8"
# modelPath = "../../neuralNet/models/citNN_comb_pool4class/citNN_comb_5band_chimQual_class4_20211009_ws15_nc200_bs01234_ne21"
# modelPath = "../../neuralNet/models/citNN_comb_4cpooled_medFilt/citNN_comb_20211024_ws15_nc200_bs01234_ne14"
# modelPath = "../../neuralNet/models/citNN_comb_4cpooled_medFilt/citNN_comb_20211024_ws15_nc200_bs023_ne18"
# modelPath = "../../neuralNet/models/citNN_comb_4c_2band/citNN_comb_20220207_ws15_nc200_bs01_ne20"
# modelPath = "../../neuralNet/models/citNN_comb_4c_2band/citNN_comb_20220207_ws15_nc200_bs0_ne25"
# modelPath = "../../neuralNet/models/citNN_comb_4c_3band/citNN_comb_20220211_ws15_nc200_bs012_ne25"
modelPath = "../../neuralNet/models/citNN_comb_4c_3band_div20/citNN_comb_20220214_ws15_nc200_bs012_ne25"
# modelPath = "../../neuralNet/models/citNN_comb_4c_3band_div20_wsSens/citNN_comb_20220222_ws9_nc200_bs012_ne30"

# fp = "../../data/testingData/citTestPatches_end_5band_medFilt_filled_102221/"
# fp = "../../data/testingData/citTestPatches_end_5band_medFilt_chimNameOnly_filled_102221/"
# fp = "../../data/testingData/citTestPatches_gsc_5band_medFilt_filled_102221/"
# fp = "G:/working_directory/Chimney-Identification-Tool/data/testingData/citTestPatches_end_namedChims_2band_20220208/"
# fp = "G:/working_directory/Chimney-Identification-Tool/data/testingData/citTestPatches_gsc_knownchims_2band_20220208/"
# fp = "G:/working_directory/Chimney-Identification-Tool/data/testingData/citTestPatches_gsc_knownchims_3band_20220214/"
# fp = "G:/working_directory/Chimney-Identification-Tool/data/testingData/citTestPatches_end_namedchims_3band_20220214/"

fp = "G:/working_directory/Chimney-Identification-Tool/data/testingData/allChimExamples_20220216/gsc/"

#%%
# trainingDataRoot1 = "../../data/testingData/allChimExamples_20220216/end/"
# trainingDataRoot2 = "../../data/testingData/allChimExamples_20220216/gsc/"

trainingDataRoot1 = "../../data/testingData/citTestPatches_end_namedchims_3band_20220215/"
trainingDataRoot2 = "../../data/testingData/citTestPatches_gsc_knownchims_3band_20220215/"

#%%

dsFull1 = citDataset(path=trainingDataRoot1, chunk=(ws,bandSet))
dsFull2 = citDataset(path=trainingDataRoot2, chunk=(ws,bandSet)) 
testDataset = dsFull1.combineDatasets(dsFull2)
testDataset.dataInfo = testDataset.dataInfo.loc[testDataset.dataInfo['labels'] == "1"]
testDataset.nclasses = nclasses

#%%
dt = date.today().strftime("%Y%m%d")
opath = "../../data/checkExpert_{}_{}.csv".format(modelPath.split("citNN_")[-1], dt)

toSave=False

#%%

loaded_model = chimNet(ws, len(bandSet), ncons, nclasses)
loaded_model.load_state_dict(torch.load( os.path.abspath(modelPath) ))

#%%
for i in range(len(testDataset)):
#%%
    
    raster, label = testDataset[i]
    
    pred = loaded_model(raster)
    pred = pred.detach().numpy()
    
    file = testDataset.getFile(i)
    
    row = pd.concat((pd.DataFrame({"file": [file], "label":[label]}),
                    pd.DataFrame({"class{}".format(v): [pred[0,v]] for v in range(pred.shape[1])})),
                    axis=1)
    
    
    if(i==0):
        outDf = row.copy()
    else:
        outDf = pd.concat( (outDf, row), axis=0 )
        
                  
    
#%%
print(modelPath)
print("thresh: {} / {}".format(np.sum(outDf['class1']>0.90), outDf.shape[0]))
print("sum: {} / {}".format(np.sum( np.logical_and(outDf['class1']>0.50, (outDf['class3']+outDf['class1'])>0.95) ), outDf.shape[0]))

#%%
if(toSave):
    outDf.to_csv(os.path.abspath(opath))





