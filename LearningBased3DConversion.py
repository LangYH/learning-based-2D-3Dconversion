# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 09:27:46 2014

@author: Administrator
"""

import numpy as np
from scipy import misc
import cv2
import os
from PIL import Image
import Filters

def computeDepthMapWithHOG( iminput, dataSetDir, depthMapDir, HOGdetector, k = 45 ):
    imNames = [ f for f in os.listdir( dataSetDir )
                    if f.endswith('.png') | f.endswith('.jpg')]
    imlist = [ os.path.join(dataSetDir,f) for f in imNames ]
    depthlist = [ os.path.join( depthMapDir, f.split('.')[0]+'_abs_smooth.png' )
                    for f in imNames ]

    if len(imlist) == 0 | len(depthlist) == 0 :
        return -1

    if len( iminput.shape ) == 3:
        im = cv2.cvtColor( iminput, cv2.COLOR_RGB2HSV )[:,:,2]
    else:
        im = iminput.copy()
    #resize objective image to the size of train images
    imSize = np.array( Image.open(imlist[0]).convert('L')).shape
    im = misc.imresize( im, imSize )


    #excute kNN search
    kIndices, distances, imlist, arrayOfDescriptors = kNNSearchWithHOG(im,imlist,HOGdetector,k)


    #read the k depthMap into 3-dimensional array kdepthMaps
    kdepthMaps = np.zeros( (imSize[0],imSize[1],len(kIndices) ) )
    for i,k in enumerate(kIndices):
        kdepthMaps[:,:,i] = np.array( Image.open(depthlist[k]) )

    #k depthMaps fused
    fusedDepthMap = fuseDepthMaps( kdepthMaps, distances )

    #cross bilateral filtered
    finalDepthMap = Filters.cross_bilateral_filter( fusedDepthMap, iminput, 10, 20, 20 )

    return finalDepthMap,kdepthMaps, fusedDepthMap, distances




def kNNSearchWithHOG( queryIm, imlist, HOGdetector, k ):
    '''excute kNN search with HOG descrptor
    queryIm: objective image
    dataSetDir:path of the dataset for kNN search
    HOGdetector: descriptor comes from opencv HOGdescripor
    k:params of kNN
    returning
    kIndices:indices of top k matching image
    imlist:the full path of training images,
    '''

    arrayOfDescriptors = computeTrainingDescriptors( imlist, HOGdetector )


    kIndices, distances = kNNSearch( HOGdetector.compute( queryIm ).T, arrayOfDescriptors, k )

    return kIndices, distances, imlist, arrayOfDescriptors

def kNNSearch( queryVector, trainSet, k ):
    print "kNN matching..."
    diffMat = trainSet - np.tile( queryVector, ( trainSet.shape[0], 1 ) )
    squareDiffMat = diffMat ** 2
    distances = ( squareDiffMat.sum( axis = 1 ) ) ** 0.5
    sortedDistanceIndices = distances.argsort()
    distances.sort()
    return sortedDistanceIndices[:k], distances[:k]

def computeTrainingDescriptors( imlist, HOGdetector ):
    print "HOGdescriptors computing..."
    trainingImages = {}
    for imfile in imlist:
        trainingImages[imfile] = cv2.cvtColor( np.array( Image.open( imfile ) ),cv2.COLOR_RGB2HSV )[:,:,2]

    sizeOfDescriptor = HOGdetector.getDescriptorSize()
    arrayOfDescriptors = np.zeros( (len(imlist), sizeOfDescriptor) )
    for i, imfile in enumerate(imlist):
        arrayOfDescriptors[i] = HOGdetector.compute( trainingImages[imfile] ).T

    return arrayOfDescriptors

def fuseDepthMaps( kdepthMaps, distances ):
    #flattenDepthMaps = kdepthMaps.reshape( kdepthMaps.shape[2], kdepthMaps.shape[0] * kdepthMaps.shape[1])
    #return np.median( flattenDepthMaps, axis = 0 ).reshape( kdepthMaps.shape[0], kdepthMaps.shape[1] )
    weights = computeWeights( distances )
    weightedkdepthMaps = np.zeros( kdepthMaps.shape )

    kdepthMaps = depthMapsNormalized( kdepthMaps )

    for i,w in enumerate(weights):
        weightedkdepthMaps[:,:,i] = w * kdepthMaps[:,:,i]

    rows,cols = kdepthMaps.shape[:2]
    filteredDepth = np.zeros( (rows,cols) )
    for i in range(rows):
        for j in range(cols):
            filteredDepth[i][j] = np.mean( weightedkdepthMaps[ i, j, :] )

    return filteredDepth

def computeWeights( distances ):
    reciprocals = np.log(1.0 / (distances + np.exp(-9)) )
    weights = reciprocals / sum( reciprocals )
    return weights

def depthMapsNormalized( kdepthMaps ):
    k = kdepthMaps.shape[2]
    for i in range(k):
        kdepthMaps[:,:,i] = kdepthMaps[:,:,i]/kdepthMaps[:,:,i].max()
    return kdepthMaps



