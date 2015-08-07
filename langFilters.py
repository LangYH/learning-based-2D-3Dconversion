# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 15:02:45 2014

@author: Administrator
"""
from numpy import *
from scipy.ndimage import filters
 
def gaussian_filter( im, sigma, wsize ):
    #im = hstack( (im[:,extendSize-1::-1], im, im[:,-1:-1-extendSize:-1] ) )
    kernel = createGaussianKernel( sigma, wsize )
    result = zeros( im.shape )
    if len(im.shape) == 3:
        for i in range(3):
            imX = filters.convolve1d( im[:,:,i], kernel, axis = 1, mode = 'reflect' )
            result[:,:,i] = filters.convolve1d( imX, kernel, axis = 0, mode = 'reflect' )
    else:
        imX = filters.convolve1d( im, kernel, axis = 1,mode = 'reflect' )
        result = filters.convolve1d( imX, kernel, axis = 0,mode = 'reflect' )
    return result
    
def bilateral_filter( im, mask_im, sigma_xy=5, sigma_value=0.8, wsize=31 ):
    #filter image im with mask mask_im(commonly the same image im )
    #compute the gaussian Kernel
    gaussianKernel = createGaussianKernel( sigma_xy, wsize )
    
    fr = wsize / 2 #radius of kernel
    im_ex = boundary_extend( im, fr )
    mask_ex = boundary_extend( mask_im, fr )
    ( rows, cols ) = im_ex.shape
    for i in arange( fr, rows - fr ):
        for j in arange( fr, cols - fr ):
            pixelValueVector = im_ex[ i, j - fr : j + fr + 1 ]
            maskValueVector = mask_ex[ i, j - fr : j + fr + 1 ]
            
            
            maskeValueKernel = exp( -( maskValueVector - mask_ex[i,j] )**2/( 2 * float(sigma_value) **2 ))
            bilateralKernel = maskeValueKernel * gaussianKernel
            bilateralKernel = bilateralKernel / sum( bilateralKernel )
            
            im_ex[i,j] = sum( pixelValueVector * bilateralKernel )
            
    for i in arange( fr, rows - fr ):
        for j in arange( fr, cols - fr ):
            pixelValueVector = im_ex[ i - fr : i + fr + 1, j ]
            maskValueVector = mask_ex[ i - fr : i + fr + 1, j ]
            
            maskValueKernel = exp( -( maskValueVector - mask_ex[i,j] )**2/( 2 * float(sigma_value) **2 ))
            bilateralKernel = maskValueKernel * gaussianKernel
            bilateralKernel = bilateralKernel / sum( bilateralKernel )
            
            im_ex[i,j] = sum( pixelValueVector * bilateralKernel )
            
    return im_ex[ fr:rows-fr, fr:cols-fr ]
    
            

    
def boundary_extend( im, fr ):
    #boundary extend before filter, fr is the radius of filter
    im_t = im.copy()
    #cols extend
    im_t = hstack( ( hstack((im_t[:, fr:0:-1 ], im_t)), im_t[:, -2:-2-fr:-1 ] ))
    #rows extend
    im_t = vstack( ( vstack(( im_t[fr:0:-1, :], im_t)), im_t[-2:-2-fr:-1, :] ) )
    return im_t
    
    
    
def createSpaceWeights( sigma, d ):
    gaussian_space_coff = -0.5/(sigma*sigma)
    space_weights = zeros( ( d, d ) )
    radius = d / 2
    for i in arange( -radius, radius + 1 ):
        for j in arange( -radius, radius + 1):
            r = sqrt( i*i + j*j )
            if r > radius:
                space_weights[ i+ radius, j+radius ] = 0.0
                continue
            space_weights[ i+ radius, j+radius ] = exp( r*r * gaussian_space_coff )
    return space_weights
#    #计算自变量distance
#    distance = abs( array( arange( -extendSize, extendSize+1 ), 'f') )
#    #正态分布概率密度
#    kernel = ( 1/( sqrt(2*pi) *sigma ) ) * exp( -distance**2/( 2 * sigma**2 ))
#    #和归一化
#    kernel = kernel / sum( kernel )
#    return kernel