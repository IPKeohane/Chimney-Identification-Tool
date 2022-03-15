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

import numpy as np
import torch
from datetime import date
from glob import glob


#%%

bandSet=[0,1,2,3,4]
ws=15
ncons=200
nclasses=4

#%%
# bandSets = [[0],(0,1,2),(0,1),(0,1,2,4),(1,2), (0,1,2,3)]
# bandSet = bandSets[i]

# modelPath = "../../neuralNet/models/citNN_end_5band_augChimDisc_ws15/citNN_end_augChimDisc_20210920_ws15_nc200_bs01234_ne9"
# modelPath = "../../neuralNet/models/citNN_end_5band_newBath_ws15/citNN_augmented_end_20210919_ws15_nc200_bs01234_3_ne9"
# modelPath = "../../neuralNet/models/citNN_comb_5band_superModel_ws15/citNN_comb_superModel_20210924_ws15_nc200_bs01234_ne11"
# modelPath = "../../neuralNet/models/citNN_end_5band_chimQual_class4_ws15/citNN_end_5band_chimQual_class4_20210927_ws15_nc200_bs01234_ne25"  
# modelPath = "../../neuralNet/models/citNN_gsc_medFilt_pool4clas/citNN_gsc_5band_chimQual_class4_20211007_ws15_nc200_bs01234_ne25"
# modelPath = "../../neuralNet/models/citNN_end_5band_nomed_3c_20211010_nc200/citNN_end_5band_nomed_3c_20211010_nc200_ws15_bs01234_ne12"
# modelPath = "../../neuralNet/models/citNN_gsc_5band_nomed_3c_20211010_nc200/citNN_gsc_5band_nomed_3c_20211010_nc200_ws15_bs01234_ne9"
# modelPath = "../../neuralNet/models/citNN_comb_5band_nomed_3c_20211010_nc200/citNN_comb_5band_nomed_3c_20211010_nc200_ws15_bs01234_ne12"
# modelPath = "../../neuralNet/models/citNN_comb_pool4class/citNN_comb_5band_chimQual_class4_20211009_ws15_nc200_bs01234_ne21"
# modelPath = "../../neuralNet/models/citNN_comb_3class_bs023/citNN_comb_5band_chimQual_class4_20211014_ws15_nc200_bs023_ne7"
# modelPath = "../../neuralNet/models/citNN_gsc_3band_3class_20211017_nc200/citNN_gsc_5band_3class_20211017_nc200_ws15_bs123_ne19"
# modelPath = "../../../../PC04_backup/working_directory/chimneyIdentificationTool/Chimney-Identification-Tool/neuralNet/models/citNN_test02_lr0.01_lt0.01_ct0.01/citNN_20210701_ws15_nc200_bs012_ne31"
# modelPath = "../../neuralNet/models/citNN_gsc_3band_3class_20211017_notaug_nc200/citNN_gsc_3band_3class_20211017_notaug_nc200_ws15_bs123_ne25"
# modelPath = "../../neuralNet/models/citNN_gsc_2band_3class_20211017_notaug_nc200/citNN_gsc_2band_3class_20211017_notaug_nc200_ws15_bs12_ne40"
# modelPath = "../../neuralNet/models/citNN_gsc_3band_3class_20211017_notaug_nc200/citNN_gsc_3band_3class_20211017_notaug_nc200_ws15_bs012_ne40"
# modelPath = "../../neuralNet/models/citNN_gsc_3band_3class_20211017_nc200/citNN_gsc_3band_3class_20211017_nc200_ws15_bs012_ne18"
# modelPath = "../../neuralNet/models/citNN_comb_3band_3class_20211017_nc200/citNN_comb_3band_3class_20211017_nc200_ws15_bs012_ne12"
# modelPath = "../../neuralNet/models/citNN_comb_1band_4class_20211018_nc200/citNN_comb_1band_4class_20211018_nc200_ws15_bs0_ne25"
# modelFolder = "../../neuralNet/models/citNN_comb_sensTest_3class_20211018_nc200/"
# modelPaths = glob(modelFolder+"*")
# modelPath = modelPaths[i]
#%%
# modelPath = "../../neuralNet/models/citNN_comb_pool4class/citNN_comb_5band_chimQual_class4_20211009_ws15_nc200_bs01234_ne21"
# modelPath = "../../neuralNet/models/citNN_comb_3cpooled_medFilt/citNN_comb_20211023_ws15_nc200_bs01234_ne8"
modelPath = "../../neuralNet/models/citNN_comb_4cpooled_medFilt20211023/citNN_comb_20211024_ws15_nc200_bs01234_ne14"
# modelPath = "../../neuralNet/models/citNN_comb_4cpooled_medFilt/citNN_comb_20211024_ws15_nc200_bs023_ne18"

#%%

dt = date.today().strftime("%Y%m%d")
opath = "../../data/gsc_chimLoc_ps2_combTrain_th090_4cpooled_b{}_{}.csv".format(len(bandSet),dt)
opathRas = "../../data/endRasters/defaultCitOut.tif"

toSave = False

#%%
# fp = "../../../../rasterStorage/Endeavor/end_ps2_5band_medFilt_filled_102321_norm.tif"
fp = "../../../../rasterStorage/GSC/gsc_ps2_5band_medFilt_filled_102321_norm.tif"

# fp = "../../../../PC04_backup/working_directory/chimneyIdentificationTool/Chimney-Identification-Tool/data/gscRasters/gsc_ps1_bpimnEslpWslp_norm_utm1m.tif"  
      
mainRaster, mainTransform, sr, _ = readRaster(fp)
mainRaster = np.where(mainRaster < -10e30,np.nan, mainRaster)
mainRaster = mainRaster[bandSet,...]


# fp = "../../data/endRasters/end_boolFilt_ws11_sp1_th1.tif"
# fp = "../../data/endRasters/end_ps2_boolFilt_ws11_sp1_th1.tif"
fp = "../../data/gscRasters/gsc_ps2_boolFilt_ws11_sp1_th1.tif"
boolRaster, boolTransform, _ , _= readRaster(fp)
boolRaster = boolRaster[0,...]

#%%
ws_full=31
windowTensors, locations = extractWindowsFromMask(boolRaster, mainRaster, 
                                      boolTransform, mainTransform,
                                      ws=ws_full, asTorch = True, verbose=0)


#%%
for i in range(len(windowTensors)):
    windowTensors[i][0,0,...] = (windowTensors[i][0,0,...]-torch.min(windowTensors[i][0,0,...]))/50.0
    

#%%
citOutputRaster = produceCitOutputRaster(windowTensors, locations, mainRaster.shape,
                                          modelPath=modelPath,
                                          ws=ws, nbands=len(bandSet), ncons=ncons, nclasses=nclasses, verbose=0)

#%%
citOutDf = produceCitOutputPoints(citOutputRaster[1,...],
                                  transform=mainTransform,
                                  min_distance=3,
                                  threshold_abs=0.90)


if(citOutputRaster.shape[0]==4):
    citOutDf3 = produceCitOutputPoints(citOutputRaster[3,...],
                                      transform=mainTransform,
                                      min_distance=3,
                                      threshold_abs=0.90)
    
    citOutDf['classOut']  = 1
    citOutDf3['classOut'] = 3
    
    citOutDf = citOutDf.append(citOutDf3)

#%%
if(toSave):
    citOutDf.to_csv(os.path.abspath(opath))
    
    writeRaster(citOutputRaster[1,...], mainTransform, oname=opathRas,
                sr=sr,ftype='GTiff')


#%%

citOutputBrick = citOutputRaster[:,locations[:,0],locations[:,1]]

if(toSave):
    np.savetxt(opath.replace("chimLoc", "citOutValues"), citOutputBrick.T, delimiter=",")






















