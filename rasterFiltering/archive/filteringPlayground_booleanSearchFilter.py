# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 13:23:31 2021

@author: Isaac
"""


import os, sys
module_path = os.path.abspath('../../src')
if module_path not in sys.path:
    sys.path.append(module_path)
    
from scriptEval import startEval
from rasterFunctions import shiftTransform, writeRaster, plotBands, readRaster
from rasterFilters import booleanDirSlope
    
import numpy as np
import pandas as pd


####################################
scrEval = startEval()
####################################

window = 5
bpiLimit=1.1
slpLimit=0.5
toSave = True


fp = os.path.abspath("../../data/endRasters/end_ps2sasq_bpi0111slpEslpN_1m_utm.tif")
mainRaster, mainTransform, sr, _ = readRaster(fp)
mainRaster = np.where(mainRaster < -10**37,np.nan,mainRaster)


boolRasterFilt = booleanDirSlope(mainRaster[2,:,:], 
                                mainRaster[1,:,:],
                                mainRaster[0,:,:],
                                window = window,
                                bpiLimit=bpiLimit,
                                slpLimit=slpLimit )
                                       
                      
boolRasTransform = shiftTransform(mainTransform,(window-1)//2,(window-1)//2)

if(toSave):
    fname = os.path.abspath("../../data/endRasters/end_ps2sasq_boolFilt_slp{}_bpi{}".format(slpLimit,bpiLimit)).replace(".", "p")
    fname = fname +".tif"
    writeRaster(boolRasterFilt, 
                boolRasTransform, 
                oname=fname,
                sr=sr,
                ftype='GTiff')


#################################
scrEval.stop()
#################################