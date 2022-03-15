# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 13:23:31 2021

@author: Isaac
"""
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from skimage import exposure

    
def getLimitsOfData(data,nodata=0,toPrint=False):
    m, n= data.shape[0], data.shape[1]
    indxs = []
    
    outside = True
    for i in range(m):
        col = data[i,:]
        if(np.max(col) == nodata):
            if(outside == False):
                indxs.append(i)
                break
            
        if(np.max(col) != nodata): 
            if(outside):
                indxs.append(i)
                outside = False
            
    outside = True        
    for j in range(n):
        col = data[:,j]
        if(np.max(col) == nodata):
            if(outside == False):
                indxs.append(j)
                break
        
        if(np.max(col) != nodata): 
            if(outside):
                indxs.append(j)
                outside = False
     
    if(toPrint): print("[top,bottom,left,right]")
    return indxs

def readRaster(fileName,multiband=True):
    fileName=os.path.abspath(fileName)
    raster = rasterio.open(fileName)
    mask = raster.dataset_mask()
    transform1 = raster.transform
    sr = raster.crs
    
    if(multiband):
        data = raster.read()
    else:
        data = raster.read(1)
    
    return data, transform1, sr, mask

def writeRaster(data, transform, oname="temp.tif",
              sr=CRS.from_epsg(32613),ftype='GTiff'):
    
    oname = os.path.abspath(oname)
    
    if(len(data.shape)==2):
        data = np.expand_dims(data, 0)
    
    with rasterio.open(
            oname,
            'w',
            driver=ftype,
            height=data.shape[1],
            width=data.shape[2],
            count=data.shape[0],
            dtype=data.dtype.name,
            crs=sr,
            transform=transform ) as outRas:

        for i in range(data.shape[0]):
            outRas.write(data[i,:,:],i+1)
            
    return

def getPatch(data,i,j,size=5):
    i = int(i)
    j = int(j)
    size = int(size)
    return data[i:i+size,j:j+size]


def breakUpRaster(data, ws=9, noChunks=4, shape=None):
    if(noChunks%2!=0): 
        print("Number of chunks not even")
        return
    
    d = (ws//2)
    M , N = data.shape[0] , data.shape[1]
    
    if(M>=N):
        nrows, ncols = noChunks//2, 2
    else:
        nrows, ncols = 2, noChunks//2
    if(shape): 
        nrows, ncols = shape[0], shape[1]
    
    dm = M//nrows
    dn = N//ncols
    
    upper_slices = [i*dm-d for i in range(nrows)]
    lower_slices = [(i+1)*dm+d for i in range(nrows)]
    left_slices = [j*dn-d for j in range(ncols)]
    right_slices = [(j+1)*dn+d for j in range(ncols)]
    
    upper_slices[0], left_slices[0] = 0, 0
    lower_slices[nrows-1], right_slices[ncols-1] = M, N
    
    coords = np.zeros((nrows,ncols,4),dtype=int)
    for m in range(nrows):
        for n in range(ncols):
            coords[m,n,:] = [upper_slices[m],left_slices[n],lower_slices[m],right_slices[n]]
    
    outChunks = []
    for m in range(coords.shape[0]):
        outChunks.append([])
        for n in range(coords.shape[1]):
            outChunks[m].append(data[coords[m,n,0]:coords[m,n,2],
                                     coords[m,n,1]:coords[m,n,3],...])
        
    return outChunks, coords    

def stitchRasterChunks(rasters):
    return np.vstack( [np.hstack(row) for row in rasters] )
    

def plotBands(data, cmap='gray'):
    if(len(data.shape)==2):
        plt.imshow( data[:,:], cmap=cmap, vmin=np.nanmin(data), vmax=np.nanmax(data) )    
        plt.colorbar()
        plt.show()
    else:
        for i in range(data.shape[0]):
            plt.imshow(data[i,:,:], cmap=cmap, vmin=np.nanmin(data[i,:,:]), vmax=np.nanmax(data[i,:,:]) )    
            plt.show()  
            
            
            
def shiftTransform(transform, dx, dy):
    '''
    Shift the raster transform by dx and dy cells.
    
    Args:
        transform (list): Rasterio transform.
        dx (int): Number of cells to shift EW, + is E.
        dy (int): Number of cells to shift NS, + is N.

    Returns:
        transform (TYPE): DESCRIPTION.

    '''
    transOut=Affine(
        transform.a,
        transform.b,
        transform.c + transform.a*dx,
        transform.d,
        transform.e,
        transform.f + transform.e*dy
        )
    
    return transOut
            
            
def getTransformIndex(i0, j0, transform1, transform2):
    '''
    (i0, j0, transform1, transform2)

    Parameters
    ----------
    i0 : INT
        row index from raster 1.
    j0 : INT
        column index from raster 1.
    transform1 : AFFINE
        Affine transform of raster 1.
    transform2 : AFFINE
        Affine transform of raster 2.

    Returns
    -------
    INT, INT
        i, j for raster 2 that corresponds to i0, j0.

    '''
    j = (j0*(transform1.a/transform2.a)) + ((transform1.c-transform2.c)/transform2.a)
    i = (i0*(transform1.e/transform2.e)) + ((transform1.f-transform2.f)/transform2.e)
    return int(np.round(i)), int(np.round(j))
            

def normalizeRaster(ip, op="default.tif", perc=0.1, toWrite=False):
    ip = os.path.abspath(ip)
    mainRaster, mainTransform, sr, _ = readRaster(ip)
    mainRaster = np.where(mainRaster < -10**37,np.nan,mainRaster)
    
    outRas = np.zeros(mainRaster.shape)
    
    for i in range(mainRaster.shape[0]):
        
        p1, p99 = np.nanpercentile(mainRaster[i,:,:], (perc, 100.0-perc))
    
        outRas[i,:,:] = exposure.rescale_intensity(
            mainRaster[i,:,:],
            in_range=(p1, p99),
            out_range=(-1.0,1.0) )
    
    
    if(toWrite):
        writeRaster(outRas, transform=mainTransform, oname=op,
                    sr=sr,ftype='GTiff')
        return
    else:
        return outRas
    
    
            
            
def getIndexFromCoords(x, y, transform):
    if(hasattr(x, "__len__")):
        i = np.array( (y-transform.f)/transform.e, dtype=np.int64 )
        j = np.array( (x-transform.c)/transform.a, dtype=np.int64 )
    else:
        i = int( (y-transform.f)/transform.e )
        j = int( (x-transform.c)/transform.a )
    return i, j
     

def getCoordsFromIndex(i, j, transform):
    x = transform.c + transform.a*j
    y = transform.f + transform.e*i
    return x, y 
            

def findNanRowsAndColumns(raster):
    raster = np.where(raster < -10e30,np.nan,raster)
    
    rows = np.argwhere(np.isnan(raster).all(axis=(0,2)))
    cols = np.argwhere(np.isnan(raster).all(axis=(0,1)))
    
    return rows[:,0], cols[:,0]

def trimPartialRowsAndColumns(raster, ws=None):
    raster = np.where(raster < -10e30,np.nan,raster)
    
    i=[]
    j=[]
    
    if(np.sum(np.isnan(raster[:,:,0]))>=6):
        j.append(0)
    elif(np.sum(np.isnan(raster[:,:,raster.shape[2]-1]))>=6):
        j.append(raster.shape[2]-1)
    
    if(np.sum(np.isnan(raster[:,0,:]))>=6):
        i.append(0)
    elif(np.sum(np.isnan(raster[:,raster.shape[1]-1,:]))>=6):
        i.append(raster.shape[1]-1) 

    return i, j
            
            
def normalizeRasterFromValues(raster,mins,maxes):
    for i in range(raster.shape[0]):
        raster[i,:,:] = (raster[i,:,:]-mins[i])/ (maxes[i]-mins[i])
    
    return raster

def cleanSquarePatchRasters(path, verbose=0, deleteAnyNan=False):
    files = glob.glob(os.path.abspath(path+"*.tif"))
    
    for n in range(len(files)):
        file=files[n]

        data, transform1, sr, _ = readRaster(file)
        
        ex=file.split("_")[-4:]
        ws = int(ex[1].replace("ws", ""))
        
        if(data.shape[1]!=ws or data.shape[2]!=ws):
            print("{} not {}x{}".format(n,ws,ws))
        
        
        nanRows, nanCols = findNanRowsAndColumns(data)
        if(nanRows.shape[0]>0):
            data = np.delete(data,nanRows,axis=1)
            if(verbose>5):
                print("n={}, nanRows={}".format(n,nanRows))
        if(nanCols.shape[0]>0):
            data = np.delete(data,nanCols,axis=2)
            if(verbose>5):
                print("n={}, nanCols={}".format(n,nanCols))

        
        if(data.shape[1]>ws or data.shape[2]>ws):
            print("{} still > {}x{} post initial Nan removal".format(n,ws,ws))
            ii, jj = trimPartialRowsAndColumns(data)
            if(len(ii)>0):
                data = np.delete(data,ii,axis=1)
            if(len(jj)>0):
                data = np.delete(data,jj,axis=2)
        
        if(data.shape[1]>ws or data.shape[2]>ws):
            print("{} manual trim to fit ws".format(n))
            data = data[:,data.shape[1]-ws:,data.shape[2]-ws:]
                
                
        if(data.shape[1]!=ws or data.shape[2]!=ws):
            print("ERROR: FINAL CHECK FAILED. {} shape {}x{} after filtering, ws={}".format(n,data.shape[1],data.shape[2],ws))
            os.remove(file)
        
        elif(np.isnan(data).any() or (data<-10e30).any()):
            print("ERROR: ANY NAN CHECK FAILED. {}".format(n))
            if(deleteAnyNan): os.remove(file)
        
        else:
            writeRaster(data, transform1, file, sr)

def findLocalMaxima(raster, threshold=-np.inf):
    
    rollStack = np.stack( (np.roll(raster,1,0), np.roll(raster,-1,0), np.roll(raster,1,1), np.roll(raster,-1,1),
                           np.roll(raster,(1,1),(0,1)), np.roll(raster,(-1,-1),(0,1)),
                           np.roll(raster,(1,-1),(0,1)), np.roll(raster,(-1,1),(0,1))), axis=0 )
    
    testStack = rollStack > np.expand_dims(raster, 0)
    
    booleanRaster = np.stack((np.invert( np.any(testStack, axis=0) ),
                              raster > threshold), axis=0)
    
    booleanRaster[:,0,:] = False
    booleanRaster[:,:,0] = False
    booleanRaster[:,booleanRaster.shape[1]-1,:] = False
    booleanRaster[:,:,booleanRaster.shape[2]-1] = False
    
    return( np.all(booleanRaster, axis=0) )

    
            
            
            
            
            
            
            
            
            