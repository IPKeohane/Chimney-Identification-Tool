# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 12:08:37 2021

@author: Isaac Keohane -- isaackeohane95@gmail.com -- github: IPKeohane
"""

import os, sys
module_path = os.path.abspath('../src')
if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
import glob
import numpy as np
import rasterio
import random

from torch.utils.data import Dataset
from rasterFunctions import readRaster, getCoordsFromIndex
from deepLearningFunctions import arrayToTensor, patchAugmentation


class citDataset(Dataset):
    
    def __init__(self, path="NA", chunk=None, files=None):
        self.path=path
        if(files is None):
            files = pd.Series(glob.glob(os.path.abspath(path+"*.tif")))
        else:
            files = pd.Series(files)
        self.dataInfo = pd.DataFrame({'files': files})
        

        self.dataInfo['labels'] = [x.split("_")[-4].split("class")[1] for x in self.dataInfo['files']]
        self.nclasses = int(len(np.unique(self.dataInfo['labels'])))
    
        ex=files.iloc[0].split("_")[-4:]
        self.ws = int(ex[1].replace("ws", ""))
        self.nbands = int(ex[2].replace("b", ""))
        if(not (chunk is None)):
           self.ws0 = int(ex[1].replace("ws", ""))
           self.nbands0 = int(ex[2].replace("b", ""))
           self.chunk = chunk
           self.ws = chunk[0]
           self.nbands = len(chunk[1],)
        else:
           self.chunk=chunk
           
        self.dataAll = None
        self.augmented = False

    def __len__(self):
        return self.dataInfo.shape[0]

    def __getitem__(self, idx):
        file = self.dataInfo.loc[idx,'files']
        data, transform1, sr, _ = readRaster(file)
        if(not (self.chunk is None)):
            ds = (self.ws0-self.ws)//2
            data = data[self.chunk[1],ds:self.ws0-ds,ds:self.ws0-ds]
        data = arrayToTensor(data)
        label = self.dataInfo.loc[idx,'labels']
        if(type(label)!=str):
            label = int(label)
        if(label not in ["E"]):
            label = int(label)
        if(self.augmented):
            data = patchAugmentation(data,
                                     self.dataInfo.loc[idx,'flip'],
                                     self.dataInfo.loc[idx,'rot'])
        return data, label
    
    def summarizeDataset(self, sendBack=False, verbose=10):
        classes, counts = np.unique(self.dataInfo['labels'], return_counts=True)
        if(verbose>5):
            for c in range(len(classes)):
                print("class {} : {}".format(classes[c],counts[c]))
        if(sendBack):
            return np.asarray((classes,counts)).T
            
    def checkDataset(self, sendBack=False):
        nfail=0
        for n in range(self.dataInfo.shape[0]):
            datasetChecks = []
            file = self.dataInfo.loc[n,'files']
            ex1 = file.split("_")[-4:]
            ws1 = int(ex1[1].replace("ws", ""))
            nbands1 = int(ex1[2].replace("b", ""))
            data, _, _, _ = readRaster(file)
            nbr = data.shape[0]
            nrow = data.shape[1]
            ncol = data.shape[2]
            if(ws1 != self.ws): datasetChecks.append("ws1")
            if(nrow != ws1): datasetChecks.append("ws2")
            if(ncol != ws1): datasetChecks.append("ws3")
            if(nbands1 != self.nbands): datasetChecks.append("nb1")
            if(nbr != nbands1): datasetChecks.append("nb2")
            if(np.isnan(data).any() or (data<-10e30).any()): datasetChecks.append("nan1")
            if(len(datasetChecks)>0):
                nfail+=1
                print("{} {} failed dataset check, codes: {}".format(n,file.split("\\")[-1],datasetChecks))
        if(nfail==0):
            print("Dataset test finished. All files pass")
            if(sendBack): return True
        else:
            print("Dataset test finished. {} files failed the test.".format(nfail))   
            if(sendBack): return False
    
    def shuffleDataset(self, equalPiles=True, resetPath=False, useAllFiles=True):
        if(equalPiles):
            if(resetPath):
                files = glob.glob(os.path.abspath(self.path+"*.tif"))
                self.dataInfo = pd.DataFrame({'files': files})
                self.dataInfo['labels'] = [int(x.split("_")[-4].split("ss")[1]) for x in self.dataInfo['files']]
            if(useAllFiles):
                if(self.dataAll is None):
                    self.dataAll = self.dataInfo.copy()
                self.dataInfo = self.dataAll.copy()
            
            classInfo = self.summarizeDataset(sendBack=True,verbose=0)
            numC1 = classInfo[1,1]
            subsets = []
            for c in range(classInfo.shape[0]):
                class1 = classInfo[c,0]
                frac = numC1*1.0 / classInfo[c,1]
                if(frac<=1.0):
                    subsets.append(self.dataInfo[self.dataInfo['labels']==class1].sample(frac=frac))
                elif(frac>=2.0):
                    print("ERROR: CLASS {} HAS FEWER THAN 1/2 THE # OF EXAMPLES AS 1".format(class1))
                    subsets.append(self.dataInfo[self.dataInfo['labels']==class1].sample(frac=1))
                else:
                    subsets.append(self.dataInfo[self.dataInfo['labels']==class1].sample(frac=1))
                    frac1 = (numC1*1.0-classInfo[c,1]) / classInfo[c,1]
                    subsets.append(self.dataInfo[self.dataInfo['labels']==class1].sample(frac=frac1))
            self.dataInfo = pd.concat(subsets).sample(frac=1).reset_index(drop=True)
        else:
            self.dataInfo = self.dataInfo.sample(frac=1).reset_index(drop=True)
            
    def subsetDataset(self, coords=None, frac=0.1, verbose=0, specifcFiles=None):
        testIdxs = []
        trainIdxs = []
        if(coords): #[x0, y0, x1, y1]
            for n in range(self.__len__()):
                data, transform, _, _ = readRaster(self.dataInfo.loc[n,'files'])
                r = (data.shape[1]/2)
                x = transform[2] + transform[0]*r
                y = transform[5] + transform[4]*r
                if(coords[0]<x and x<coords[2] and coords[1]<y and y<coords[3]):
                    if(verbose>9): print("{} {} in coords".format(n,self.dataInfo.loc[n,'files'].split("\\")[-1]))
                    testIdxs.append(n)
                else:
                    if(verbose>9): print("{} {} outside coords".format(n,self.dataInfo.loc[n,'files'].split("\\")[-1]))
                    trainIdxs.append(n)
        else:
            for c in range(np.max(self.dataInfo['labels'])+1):
                subset = self.dataInfo[self.dataInfo['labels']==c]
                testIdxs += list(subset.sample(frac=frac).index)
                for n in subset.index:
                    if(n not in testIdxs): trainIdxs.append(n)
        return trainIdxs, testIdxs 
    
    def removeItems(self, files, verbose=0):
        shapeCheck = self.dataInfo.shape[0]
        if(type(files)==str):
            files = [files]
        for file in files:
            
            shapeCheck1 = self.dataInfo.shape[0]
            
            self.dataInfo = self.dataInfo[(self.dataInfo['files']!=file) |
                                          (self.dataInfo['flip'] !=0) |
                                          (self.dataInfo['rot']  !=0) ] 
            
            if(verbose>6): 
                if(shapeCheck1==self.dataInfo.shape[0]): print("NO ITEM REMOVED {}".format(file))
        if(verbose>3):
            if(shapeCheck-self.dataInfo.shape[0] != len(files)):
                print("{} ITEMS INPUT, ONLY {} REMOVED".format(len(files), shapeCheck-self.dataInfo.shape[0]))
    
    
    
    def augmentDataset(self):
        if(self.augmented):
            print("ALREADY AUGEMENTED")
            return 
        
        modifiers=np.array([[0,90],[0,180],[0,270],
                           [1,0],[1,90],[1,180],[1,270]])
        self.augmented = True
        self.resetIndexes()
        self.dataInfo["flip"]=0
        self.dataInfo["rot"]=0
        
        nfiles = len(self)
        for n in range(nfiles):
            file = self.dataInfo.iloc[n,0]
            label = self.dataInfo.iloc[n,1]
            newRow = pd.DataFrame({'files':file,
                                   'labels':label,
                                   'flip':modifiers[:,0],
                                   'rot':modifiers[:,1]})
            self.dataInfo = self.dataInfo.append(newRow)
        self.resetIndexes()
        
    def sortDataset(self):
        self.dataInfo=self.dataInfo.sort_values("labels")
        self.resetIndexes()
    
    def resetIndexes(self):
        self.dataInfo = self.dataInfo.reset_index(drop=True)
    
    def getCenter(self, idx):
        file = self.dataInfo.loc[idx,'files']
        data, transform1, sr, _ = readRaster(file)
        ws = data.shape[2]
        r = ws//2
        return getCoordsFromIndex(r, r, transform1)
    
    def combineDatasets(self, dsNew):
        self.dataInfo = pd.concat( [self.dataInfo, dsNew.dataInfo], ignore_index=True)
        return(self)
    
    def getFile(self, idx):
        return self.dataInfo.loc[idx,'files']
        
        
# path = "../data/trainingData/citTrainPatches_gsc_c3_ws15_b3/"
# dsFull = citDataset(path)      
        
        
        
        
        
        
        
        
        
        
        
        
            
 
