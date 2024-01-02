# # -*- coding: utf-8 -*-
# """
# Created on Wed Jun 23 13:04:00 2021

# @author: Isaac
# """

# import os, sys
# module_path = os.path.abspath('../../src')
# if module_path not in sys.path:
#     sys.path.append(module_path)
# from scriptEval import startEval

# from producingCitOutputs import produceCitOutputPoints, produceCitOutputRaster
# from rasterFunctions import readRaster, writeRaster
# from deepLearningFunctions import extractWindowsFromMask

# import numpy as np
# import torch
# from datetime import date
# from glob import glob


# #%%

# bandSet=[0,1,2]
# ws=15
# ncons=200
# nclasses=4

# #%%

# modelPath = "../../neuralNet/models/"

# #%%

# dt = date.today().strftime("%Y%m%d")
# opath = "../../data/gsc_chimLoc_ps2_combTrain_th090_4cpooled_b{}_{}.csv".format(len(bandSet),dt)
# opathRas = "../../data/endRasters/defaultCitOut.tif"

# toSave = False

# #%%
# # fp = "../../../../rasterStorage/Endeavor/end_ps2_5band_medFilt_filled_102321_norm.tif"
    
# mainRaster, mainTransform, sr, _ = readRaster(fp)
# mainRaster = np.where(mainRaster < -10e30,np.nan, mainRaster)
# mainRaster = mainRaster[bandSet,...]


# # fp = "../../data/endRasters/end_boolFilt_ws11_sp1_th1.tif"
# # fp = "../../data/endRasters/end_ps2_boolFilt_ws11_sp1_th1.tif"
# fp = "../../data/gscRasters/gsc_ps2_boolFilt_ws11_sp1_th1.tif"
# boolRaster, boolTransform, _ , _= readRaster(fp)
# boolRaster = boolRaster[0,...]

# #%%
# ws_full=31
# windowTensors, locations = extractWindowsFromMask(boolRaster, mainRaster, 
#                                       boolTransform, mainTransform,
#                                       ws=ws_full, asTorch = True, verbose=0)


# #%%
# for i in range(len(windowTensors)):
#     windowTensors[i][0,0,...] = (windowTensors[i][0,0,...]-torch.min(windowTensors[i][0,0,...]))/50.0
    

# #%%
# citOutputRaster = produceCitOutputRaster(windowTensors, locations, mainRaster.shape,
#                                           modelPath=modelPath,
#                                           ws=ws, nbands=len(bandSet), ncons=ncons, nclasses=nclasses, verbose=0)

# #%%
# citOutDf = produceCitOutputPoints(citOutputRaster[1,...],
#                                   transform=mainTransform,
#                                   min_distance=3,
#                                   threshold_abs=0.90)


# if(citOutputRaster.shape[0]==4):
#     citOutDf3 = produceCitOutputPoints(citOutputRaster[3,...],
#                                       transform=mainTransform,
#                                       min_distance=3,
#                                       threshold_abs=0.90)
    
#     citOutDf['classOut']  = 1
#     citOutDf3['classOut'] = 3
    
#     citOutDf = citOutDf.append(citOutDf3)

# #%%
# if(toSave):
#     citOutDf.to_csv(os.path.abspath(opath))
    
#     writeRaster(citOutputRaster[1,...], mainTransform, oname=opathRas,
#                 sr=sr,ftype='GTiff')


# #%%

# citOutputBrick = citOutputRaster[:,locations[:,0],locations[:,1]]

# if(toSave):
#     np.savetxt(opath.replace("chimLoc", "citOutValues"), citOutputBrick.T, delimiter=",")






















