# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:21:56 2021

@author: Isaac Keohane -- isaackeohane95@gmail.com -- github: IPKeohane
"""


import os, sys
module_path = os.path.abspath('../../src')
if module_path not in sys.path:
    sys.path.append(module_path)
from scriptEval import startEval

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%

knownChims = pd.read_csv( os.path.abspath("../../data/gsc_chimneys_ps1.txt") )
knownChims = knownChims[['x_utm', "y_utm"]].to_numpy()


def getChimProximityStats(prodChims, knownChims):
    xmesh = np.meshgrid(prodChims[:,0], knownChims[:,0])
    ymesh = np.meshgrid(prodChims[:,1], knownChims[:,1])
    
    d = np.sqrt( (xmesh[0] - xmesh[1])**2 + (ymesh[0] - ymesh[1])**2 )
    
    return np.min(d, axis=0), np.min(d, axis=1)






#%%
prodChims = pd.read_csv( os.path.abspath("../../data/gsc_ps1_nonMlChimLocations_ws9sp3th1.csv") )
prodChims = prodChims[['x_utm', "y_utm"]].to_numpy()

prodProx, knownProx = getChimProximityStats(prodChims, knownChims)

fig1, ax1 = plt.subplots(nrows=2, ncols=2, sharex='row')

ax1[0,0].hist(prodProx)
ax1[1,0].hist(knownProx)


ax1[0,0].set_xlabel("Distance of each produced chim to nearest known chim")
ax1[1,0].set_xlabel("Distance of each known chim to nearest produced chim")

#%%
prodChims = pd.read_csv(
    os.path.abspath("../../data/localMaximaOutputs/citOut_3bandOut_3classWs15_30ep_0p9thr.csv"),
    header=None)
prodChims = prodChims[[0, 1]].to_numpy()

prodProx, knownProx = getChimProximityStats(prodChims, knownChims)


ax1[0,1].hist(prodProx)
ax1[1,1].hist(knownProx, bins=20)


ax1[0,1].set_xlabel("Distance of each produced chim to nearest known chim")
ax1[1,1].set_xlabel("Distance of each known chim to nearest produced chim")







