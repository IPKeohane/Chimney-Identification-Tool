# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 11:46:11 2021

@author: Isaac
"""


import os, sys
module_path = os.path.abspath('../../src')
if module_path not in sys.path:
    sys.path.append(module_path)

from rasterFunctions import shiftTransform, writeRaster, plotBands, readRaster
from citDataset import citDataset
import numpy as np
import glob

from skimage.transform import rescale
import matplotlib.pyplot as plt

#%%
dummyRas = np.random.rand(50,50)-0.5

#%%
#plotBands(dummyRas)

#%%

dummy = np.zeros((11,11))
for i in range(dummy.shape[0]):
    for j in range(dummy.shape[1]):
        x = (5.01-j)
        y = (5.01-i)
        dummy[i,j] = (1.14**(-1*np.sqrt(x**2+y**2)**2))
dummy=dummy-np.min(dummy)
dummy=dummy/np.max(dummy)


patch1 = np.zeros((25,25))
patch1[7:18,7:18]=dummy

patch2 = np.zeros((25,25))
patch2[2:23,2:23]=rescale(dummy,21/11)*(21/11)

#%%
def rescaleChim(patch, targetValue=50):
    patch = patch-np.min(patch)
    ws = patch.shape[0]
    if(patch.shape[1]!=ws): print("NOT SQUARE")
    
    ci = (ws//2)
    
    peak = np.max(patch[ci-2:ci+3,ci-2:ci+3])
    
    if(peak<2):
        out = rescale(patch,targetValue)*(targetValue)
    else:
        out = rescale(patch,targetValue/peak)*(targetValue/peak)
    
    if(out.shape[0]%2 == 0):
        out = rescale(out, (out.shape[0]+1)/out.shape[0])*(out.shape[0]+1)/out.shape[0]
        
        
    return out

#%%
test1 = rescaleChim(patch1)
test2 = rescaleChim(patch2)

#%%
def plotSideView(patch,axis="x"):
    if(axis=="y"):
        x=np.arange(patch.shape[0])
        y=patch[:,patch.shape[0]//2]
        
    if(axis=="x"):
        x=np.arange(patch.shape[1])
        y=patch[patch.shape[1]//2,:]
    
    plt.plot(x,y)
    #plt.show()

#%%
def stackAndTrimPatches(arrayList, normalize=True):
    dims = []
    for patch in arrayList:
        dims.append(patch.shape[0])
        if(patch.shape[0]!=patch.shape[1]): print("NOT SQUARE IN STACK")
        
    dims = np.array(dims, dtype=int)
    m = np.min(dims)
    outputStack = np.zeros((len(arrayList),m,m))
    
    i = 0
    for patch in arrayList:
        n0 = patch.shape[0]
        d = (n0-m)//2
        
        if(normalize):
            outputStack[i,...] = patch[d:n0-d,d:n0-d] / np.max(patch[d:n0-d,d:n0-d])
        else:
            outputStack[i,...] = patch[d:n0-d,d:n0-d]
        
        i+=1
        
    return outputStack

#%%
def ringAverage(patch, r):
    m = patch.shape[0]
    
    mask = np.zeros(patch.shape)
    mask[r,r:m-r-1] = 1
    mask[m-r-1,r:m-r] = 1
    mask[r:m-r,r] = 1
    mask[r:m-r,m-r-1] = 1
    
    if(r==(m//2+1)):
        mask[r,r]=1
    
    return(np.average(patch,weights=mask))
    
#%%
def stackPatchScaleDataset(patchDataset, targetValue=50, normalize=True):
    arrayList = []
    for i in range(len(patchDataset)):
        patch = patchDataset[i][0].numpy()[0,0,...]
        arrayList.append( rescaleChim(patch, targetValue) )
    
    return stackAndTrimPatches(arrayList, normalize=normalize)


#%%
chimBthDataset = citDataset("../../data/chimBthPatches/",chunk=(21,[0]))

edgeBthDataset = citDataset("../../data/edgeBthPatches/",chunk=(21,[0]))

chimBthDatasetGsc = citDataset("../../data/chimBthPatchesGsc/",chunk=(21,[0]))

#%%

combDataset = citDataset("../../data/bthOnlyPatches_end/",chunk=(21,[0]))
dummy, _ = combDataset.subsetDataset(frac=0.9)
combDataset = citDataset(chunk=(21,[0]), files=combDataset.dataInfo['files'][dummy])
combDataset.resetIndexes()

files= glob.glob(os.path.abspath("../../data/trainingData/citTrainPatches_gsc_c3_ws31_b5/class1*.tif"))
chimDatasetGsc = citDataset(files=files, chunk=(15,[0]))


#%% 
norm = False
tgtValue = 58

outputStackChimEnd = stackPatchScaleDataset(chimBthDataset, tgtValue, normalize=norm)
outputStackEdge = stackPatchScaleDataset(edgeBthDataset, tgtValue, normalize=norm)
outputStackChimGsc = stackPatchScaleDataset(chimBthDatasetGsc, tgtValue, normalize=norm)
outputStackComb = stackPatchScaleDataset(combDataset, tgtValue, normalize=norm)

#%%
avgChimEnd = np.mean(outputStackChimEnd,axis=0)
r = avgChimEnd.shape[0]//2
avgChimVectorEnd = np.zeros(r)
for i in range(r):
    avgChimVectorEnd[i] = ringAverage(avgChimEnd, i)
    
avgChimGsc = np.mean(outputStackChimGsc,axis=0)
r = avgChimGsc.shape[0]//2
avgChimVectorGsc = np.zeros(r)
for i in range(r):
    avgChimVectorGsc[i] = ringAverage(avgChimGsc, i)


#%%
plt.plot( (14-np.arange(len(avgChimVectorEnd))), avgChimVectorEnd)





