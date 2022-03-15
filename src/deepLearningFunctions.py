# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:36:52 2021

@author: Isaac Keohane -- isaackeohane95@gmail.com -- github: IPKeohane
"""

import numpy as np
import torch
import os
import glob
import pandas as pd
from rasterFunctions import getTransformIndex
from skimage.transform import rescale, rotate

def arrayToTensor(raster, normalize=False):
    if(len(raster.shape)>4 or 0>len(raster.shape)):
        print("ERROR: arrayToTensor received weird dimensions {}".format(raster.shape))
        return
    for i in range(4-len(raster.shape)):
        raster = np.expand_dims(raster,0)
    if(normalize):
        raster = raster - np.expand_dims(np.min(raster,axis=(0,2,3)),(0,2,3)) # subtract min from each band
        raster = raster / np.expand_dims(np.max(raster,axis=(0,2,3)),(0,2,3))
    return torch.from_numpy( np.array(raster, dtype=np.float32) )



def extractWindowsFromMask(boolRaster, raster, boolTransform, transform, 
                           ws=15, asTorch = True, verbose=0):
    if(np.min(boolRaster)!=0 or np.max(boolRaster)!=1): # check that the boolean mask is in 0 and 1 
        print("ERROR: BOOLEAN MASK NOT BINARY FORMATTED")
        return
    if(len(raster.shape)==2):
        raster = np.expand_dims(raster,0)
        if(verbose>0): print("RESHAPED HxW RASTER TO BxHxW")
    indexes = np.argwhere(boolRaster) # get the indexes of the desired cells
    dc = (ws-1)//2
    imin, jmin, imax, jmax = dc, dc, raster.shape[1]-dc-1, raster.shape[2]-dc-1
    outList = []
    locations = np.array(np.zeros(shape=(0,2)),dtype=np.int32)
    for n in range(indexes.shape[0]):
        i , j = getTransformIndex(indexes[n,0],indexes[n,1],boolTransform,transform)
        if(i<imin or j<jmin or i>imax or j>jmax):
            if(verbose>8): print("SKPIPPED {},{} - OUT OF BOUNDS".format(i,j))
            continue
        
        window = raster[:,i-dc:i+dc+1,j-dc:j+dc+1]
        
        if(np.isnan(window).any()):
            if(verbose>8): print("SKPIPPED {},{} - CONTAINS NAN".format(i,j))
            continue
        if(asTorch):
            outList.append( arrayToTensor(window,normalize=False) )
            locations = np.vstack( (locations, (i,j)) )
        else:
            outList.append(window)
            locations = np.vstack( (locations, (i,j)) )
    
    return outList, locations


def nnOutputRes(f_list, s_list, n0):
    n=n0
    for i in range(len(f_list)):
        n = (n-f_list[i])/s_list[i] + 1
    
    return n

def printTensorClasses(output, label):
    numClasses = label.shape[1]
    gr = int(torch.argmax(label))
    
    message = "Ref. class = %i, model output:" % (gr)
    for n in range(numClasses):
        message += " %i = %.3f |" % (n,float(output[0,n]))
        
    print(message)
    
def labelTensor(c, nclasses):
    label = torch.zeros((1,nclasses), dtype=torch.float32)
    label[0,c]=1.0
    return label

def bsFromTag(bsTag):
    bs = [int(b) for b in bsTag]
    return bs    
    
def parseFilenameTags(filename, flags=["ws","nc","bs"]):
    fileInfoTags = [filename.split(("_"+f))[1].split("_")[0] for f in flags]
    return fileInfoTags

def loadOutputDirectory(path, nameTag="*", flags=["ws","nc","bs"], colTypes=None):
    path = os.path.abspath(path)
    files = glob.glob(  os.path.join(path,nameTag)  )
    for i in range(len(files)):
        file = files[i]
        fileInfoTags = parseFilenameTags(file,flags)
        row = pd.DataFrame(data=[[file]+fileInfoTags], columns=["file"]+flags)
        if(i==0):
            fileInfoDf = row
        else:
            fileInfoDf = fileInfoDf.append(row, ignore_index=True)   
    if(colTypes is not None):
       fileInfoDf = fileInfoDf.astype({flags[i]: colTypes[i] for i in range(len(flags))})
    return fileInfoDf

def rescaleChimTensor(patch, targetValue=50):
    tensorType = patch.dtype
    patch = patch.numpy()[0,0,...]
    
    patch = patch-np.min(patch)
    ws = patch.shape[0]
    if(patch.shape[1]!=ws): print("NOT SQUARE")
    ci = (ws//2)
    peak = np.max(patch[ci-2:ci+3,ci-2:ci+3])
    
    if(peak<1):
        out = rescale(patch,targetValue)*(targetValue)
    else:
        out = rescale(patch,targetValue/peak)*(targetValue/peak)
    
    if(out.shape[0]%2 == 0):
        out = rescale(out, (out.shape[0]+1)/out.shape[0])*(out.shape[0]+1)/out.shape[0]
    
    out = torch.tensor(out, dtype=tensorType)
    out = out.unsqueeze(0)
    return out.unsqueeze(0)

def patchRescaleForNn(patch, targetValue=50, normalize=True, ws=21):
    patch = rescaleChimTensor(patch, targetValue)
    n0 = patch.shape[2]
    d = (n0-ws)//2
    patch1 =  patch[:,:,d:n0-d,d:n0-d]
    
    if(normalize):
        patch1 = patch1-torch.min(patch1)
        return patch1/ torch.max(patch1)
    else:
        return patch[:,:,d:n0-d,d:n0-d]

def patchAugmentation(patch, flip=0, rot=0):
    tens = False
    if(torch.is_tensor(patch)):
        tens=True
        dtype = patch.dtype
        ndims = len(patch.shape)
        if(ndims==4):
            patch = patch.numpy()[0,...]
    if(len(patch.shape))==2:
        if(flip==1):
            patch = np.flipud(patch)
    elif(len(patch.shape)==3):
        for i in range(patch.shape[0]):
            layer = patch[i,...]
            if(flip==1):
                layer = np.flipud(layer)
            patch[i,...] = rotate(layer, rot)    
    else:
        print("WEIRD DIMENSIONS IN PATCH AUGMENTATION")
        return
    if(tens):
        return torch.tensor(patch, dtype=dtype).unsqueeze(0)
    else:
        return patch

    
   
    
    
    
    
    
