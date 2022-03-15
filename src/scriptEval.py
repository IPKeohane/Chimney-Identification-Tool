# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:00:29 2021

@author: Isaac
"""

import time
import tracemalloc
import datetime

class startEval:
    
    def __init__(self):
        self.startTime = time.time()
        tracemalloc.start()
        
    def stop(self):
        elapsedT = time.time() - self.startTime
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        s1 = "Runtime: {}".format(str(datetime.timedelta(seconds=round(elapsedT))))
        s2 = "Peak memory usage: {}".format(self.formatMem(peak))
        s3 = "Ending memory usage: {}".format(self.formatMem(current))
        
        print("{} -- {} -- {}".format(s1,s2,s3))
   
        
   
    
    def formatMem(self,num):
        num = num/(10**6)
        if(num<1):
            return("{:.3f} MB".format(num))
        elif(num<1024):
            return("{:,} MB".format(round(num)))
        else:
            return("{:,} GB".format(round(num/1024)))
    