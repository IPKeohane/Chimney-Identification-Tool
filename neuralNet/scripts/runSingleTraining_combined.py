# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:55:30 2021

@author: Isaac
"""
import os, sys
module_path = os.path.abspath('../../src')
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd

from citDataset import citDataset
from scriptEval import startEval
from trainChimModel import trainChimModel
from datetime import date

####################################
scrEval = startEval()
####################################

#%%

windows = [9,11,13]

#%%
for ws in windows:
#%%
    #%%
    # ws = 15
    ncons = 200
    bandSet = (0,1,2)
    
    nn_lr = 0.1
    gamma = 0.1
    milestones = [1,16,22]
    
    nn_lossThresh = 0.005
    nn_convThresh = 0.05
    maxEpoch = 30
    
    verbose = 9
    location = "comb"
    
    #gsc_psbox1 = [620918, 231295, 622554,  232931]
    end_ps2sasq_ll = [-129.075, 47.9923, -129.0605, 48.0018]
    gsc_ps2_utm = [621175, 231591, 622089, 232798]
    
    bbox1 = end_ps2sasq_ll
    bbox2 = gsc_ps2_utm
    
    
    #%%
    modelFolder = "citNN_comb_4c_3band_div20_wsSens/"
    #modelFolder = "citNN_default/"
    
    modelRoot = os.path.abspath("../models/")
    outputStatRoot = os.path.abspath("../outputStats/")
    
    #%%
    
    # trainingDataRoot1 = "../../data/trainingData/citTrainPatches_end_3band_20220210/"
    # trainingDataRoot2 = "../../data/trainingData/citTrainPatches_gsc_3band_20220210/"
    
    trainingDataRoot1 = "../../data/trainingData/citTrainPatches_end_3band_20220214/"
    trainingDataRoot2 = "../../data/trainingData/citTrainPatches_gsc_3band_20220214/"
    
    #%%
    
    modelFolder = os.path.abspath( os.path.join(modelRoot, modelFolder) )
    if(not os.path.isdir(modelFolder)):
        os.makedirs(modelFolder)
    
    #%%
    nbands=len(bandSet)
    
    bsCode = "".join( [str(b) for b in bandSet] )
    dt = date.today().strftime("%Y%m%d")
    modelPath = os.path.join(modelFolder, "citNN_{}_{}_ws{}_nc{}_bs{}".format(location, dt, ws,ncons,bsCode))
    if(verbose>7): print(modelPath)
    
    #%%
    dsFull1 = citDataset(path=trainingDataRoot1, chunk=(ws,bandSet))
    dsFull2 = citDataset(path=trainingDataRoot2, chunk=(ws,bandSet)) 
    
    trainIdxs1, testIdxs1 = dsFull1.subsetDataset(coords=bbox1, verbose=verbose)
    trainIdxs2, testIdxs2 = dsFull2.subsetDataset(coords=bbox2, verbose=verbose)
    
    dsTrain1 = citDataset(chunk=(ws,bandSet), files=dsFull1.dataInfo['files'][trainIdxs1])
    dsTest1 = citDataset(chunk=(ws,bandSet), files=dsFull1.dataInfo['files'][testIdxs1])
    dsTrain2 = citDataset(chunk=(ws,bandSet), files=dsFull2.dataInfo['files'][trainIdxs2])
    dsTest2 = citDataset(chunk=(ws,bandSet), files=dsFull2.dataInfo['files'][testIdxs2])
    
    dsTest1.resetIndexes()
    dsTrain1.resetIndexes()
    dsTest2.resetIndexes()
    dsTrain2.resetIndexes()
    

    #%%
    dsTrain = dsTrain1.combineDatasets(dsTrain2)
    dsTest = dsTest1.combineDatasets(dsTest2)
    
    #%%
    dsTest.sortDataset()
    dsTrain.augmentDataset()
    
    #%%
    chimDataEnd = pd.read_csv("../../data/end_clagueNamedchims.txt")
    chimDataGsc = pd.read_csv("../../data/gsc_ALeeAchims.txt")
    
    
    endKnownFiles = [os.path.abspath("{}class1_ws15_b3_n{}.tif".format(trainingDataRoot1, OID)) for OID in chimDataEnd['OBJECTID']]
    gscKnownFiles = [os.path.abspath("{}class1_ws15_b3_n{}.tif".format(trainingDataRoot2, OID)) for OID in chimDataGsc['OBJECTID']]
    
    dsTrain.removeItems(endKnownFiles, verbose=5)
    dsTrain.removeItems(gscKnownFiles, verbose=5)
    
    #%%
    
    lossOverTime, modelPath = trainChimModel(dsTrain, dsTest, ncons=ncons, modelPath=modelPath,
            equalPiles=True, resetPath=False, useAllFiles=True, lr=nn_lr, maxEpoch=maxEpoch,
            lossThreshold=nn_lossThresh, convergeThreshold=nn_convThresh, verbose=verbose,
            gamma=gamma, milestones=milestones)
    if(verbose>1): print("Finished training, # of epochs = {}".format(len(lossOverTime)))
    
    
   
    
#%%
#################################
scrEval.stop()
#################################
