# -*- coding: utf-8 -*-
"""
Train a chimnet using a citdataset

Created on Mon Mar  1 14:53:12 2021

@author: Isaac Keohane -- isaackeohane95@gmail.com -- github: IPKeohane
"""

import os, sys
module_path = os.path.abspath('../../src')
if module_path not in sys.path:
    sys.path.append(module_path)
    
from scriptEval import startEval
from deepLearningFunctions import printTensorClasses, labelTensor
from chimNet import chimNet

import glob
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.optim as optim


#%%
def trainChimModel(dsTrain, dsTest, ncons=50, modelPath="../models/defaultModel",
                   equalPiles=True, resetPath=False, useAllFiles=True, lr=0.01, 
                   lossThreshold = 0.01, convergeThreshold=0.01, maxEpoch=50, milestones=[1,5,10],
                   gamma=1.0, verbose=0):
    ####################################
    scrEval = startEval()
    ####################################
    
#%%    
    modelPath = os.path.abspath(modelPath)
    
    #%%
    #load the model
    model = chimNet(dsTrain.ws, dsTrain.nbands, ncons=ncons, nclasses=dsTrain.nclasses)
    # create the optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    if(verbose>5): 
        print("before training on each raster using epochs")
        for j in range(len(dsTest)):
            printTensorClasses( model(dsTest[j][0]), labelTensor(dsTest[j][1], dsTest.nclasses) )
    
    
    #%%
    lossOverTime = []
    epoch = 0
    avgLoss = 1
    movingAvg = 1
    window = [10,5,1]
    
    while(avgLoss > lossThreshold or movingAvg > convergeThreshold):
        epoch += 1
        
        dsTrain.shuffleDataset(equalPiles=equalPiles, resetPath=resetPath, useAllFiles=useAllFiles)
        
        avgLoss = 0
        for i in range(len(dsTrain)):
            raster, label = dsTrain[i]
            label = labelTensor(label, dsTrain.nclasses)
            
            optimizer.zero_grad() # reset the gradients in the NN
        
            output = model(raster) # run the model on the current raster
        
            loss_fn = nn.MSELoss() # define the loss function
        
            loss = loss_fn(output, label) # calculate the loss
            
            avgLoss += float(loss) # collect the loss for raster i
        
            loss.backward() # back-propogate to fill in the gradient values in the NN
        
            optimizer.step() # update the NN node weights using the new gradients and lr refined in the optimizer
            
        scheduler.step()
        
        avgLoss = avgLoss/len(dsTrain)
            
        if(verbose>7):
            for j in range(len(dsTest)):
                printTensorClasses(model(dsTest[j][0]), labelTensor(dsTest[j][1], dsTest.nclasses))
        if(verbose>1): print("average loss for epoch %i: %.6f" % (epoch, avgLoss))
        lossOverTime.append( avgLoss )
        
        window[0:3] = window[1:]+[avgLoss] 
        movingAvg = np.abs(1-(np.mean(window)/avgLoss))
        if(verbose>1): print("rolling avg of loss for epoch %i: %.6f" % (epoch, movingAvg))
        
        modelPath1 = modelPath + "_ne{}".format(len(lossOverTime))
        torch.save(model.state_dict(), modelPath1)
        
        if(epoch>=maxEpoch):
            print("TRAINING EXITED.  CONVERGENCE NOT REACHED BY EPOCH {}".format(epoch))
            break
        
    
    #%%
    # load the saved model state
    loaded_model =  chimNet(dsTrain.ws, dsTrain.nbands, ncons=ncons, nclasses=dsTrain.nclasses)
    loaded_model.load_state_dict(torch.load(modelPath1))
    
    #%%
    if(verbose>0):
        print("After training on each raster using epochs")
        for j in range(len(dsTest)):
            printTensorClasses(model(dsTest[j][0]), labelTensor(dsTest[j][1], dsTest.nclasses))

#%%
    #################################
    scrEval.stop()
    #################################
    
    return lossOverTime, modelPath










