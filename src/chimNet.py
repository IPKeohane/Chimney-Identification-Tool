# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 16:19:31 2021

@author: Isaac Keohane -- isaackeohane95@gmail.com -- github: IPKeohane
"""
import os, sys
module_path = os.path.abspath('../../src')
if module_path not in sys.path:
    sys.path.append(module_path)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from deepLearningFunctions import arrayToTensor

class chimNet(nn.Module):

    def __init__(self, ws, nbands, ncons=150, nclasses=3):
        super(chimNet, self).__init__()
        # self.<layer name> = nn.Conv2d(num. layers in previous layer,
        #                               num. of conv. filters to use,
        #                               dimensions of conv. filter    )
        # if using the format from 
        self.conv1 = nn.Conv2d(nbands, ncons, 3, padding=1)
        self.conv2 = nn.Conv2d(ncons, ncons, 3, padding=1)

        self.ws = ws
        self.ncons = ncons
        
        testRas = arrayToTensor(np.zeros((1,nbands,ws,ws)))
        x = F.relu(self.conv1(testRas))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.conv2(x))
        if(self.ws>8):
            if(self.ws<22):
                x = F.max_pool2d(x, 3, 1)
            else:
                x = F.max_pool2d(x, 3, 2)
        
        
        if(self.ws==100):
            self.conv1 = nn.Conv2d(nbands, 200, 5, 2, padding=2)
            self.conv2 = nn.Conv2d(200, 350, 3, 1,  padding=1)
            self.conv3 = nn.Conv2d(350, 350, 3, 1, padding=1)
            self.conv4 = nn.Conv2d(350, 350, 3, 1, padding=1)
            self.conv5 = nn.Conv2d(350, 350, 3, 1, padding=1)
            x = F.relu(self.conv1(testRas))
            x = F.max_pool2d(x, 3, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 3, 2)
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = F.relu(self.conv5(x))
            x = F.max_pool2d(x, 3, 2)
            
            
        nfc1 = self.num_flat_features(x)
        self.n_fc_layers = math.ceil(math.log(nfc1/3,4))

        #print("ws={} ncon={} --> nfc1={} n_fc_layers={}".format(ws,ncons,nfc1,self.n_fc_layers))
        
        self.fc0 = nn.Linear(12,nclasses)
        if(self.n_fc_layers<3):
            self.fc1 = nn.Linear(nfc1,12)
        else:
            self.fc1 = nn.Linear(48,12)
            if(self.n_fc_layers<4):
                self.fc2 = nn.Linear(nfc1,48)
            else:
                self.fc2 = nn.Linear(192,48)
                if(self.n_fc_layers<5):
                    self.fc3 = nn.Linear(nfc1,192)
                else:
                    self.fc3 = nn.Linear(768,192)
                    if(self.n_fc_layers<6):
                        self.fc4 = nn.Linear(nfc1,768)
                    else:
                        self.fc4 = nn.Linear(3072,768)
                        if(self.n_fc_layers<7):
                            self.fc5 = nn.Linear(nfc1,3072)
                        else:
                            self.fc5 = nn.Linear(12288,3072)
                            if(self.n_fc_layers<8):
                                self.fc6 = nn.Linear(nfc1,12288)
                            else:
                                self.fc6 = nn.Linear(49152,12288)
                                if(self.n_fc_layers<9):
                                    self.fc7 = nn.Linear(nfc1,49152)
                                else:
                                    self.fc7 = nn.Linear(49152*4,49152)
        
    def forward(self, x):
        
        if(self.ws==100):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 3, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 3, 2)
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = F.relu(self.conv5(x))
            x = F.max_pool2d(x, 3, 2)
            
        else:
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 3, 2)
            x = F.relu(self.conv2(x))
            if(self.ws>8):
                if(self.ws<22):
                    x = F.max_pool2d(x, 3, 1)
                else:
                    x = F.max_pool2d(x, 3, 2)
        
            
        #flatten so we can use fully connected
        x = x.view(-1, self.num_flat_features(x))
        
        if(self.n_fc_layers>7):
            x = F.relu(self.fc7(x))
        if(self.n_fc_layers>6):
            x = F.relu(self.fc6(x))
        if(self.n_fc_layers>5):
            x = F.relu(self.fc5(x))
        if(self.n_fc_layers>4):
            x = F.relu(self.fc4(x))
        if(self.n_fc_layers>3):
            x = F.relu(self.fc3(x))
        if(self.n_fc_layers>2):
            x = F.relu(self.fc2(x))
        if(self.n_fc_layers>1):
            x = F.relu(self.fc1(x))
        x = self.fc0(x)
        
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
