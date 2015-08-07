# -*- coding: utf-8 -*-
"""
Created on Wed Oct 01 14:04:19 2014

@author: Administrator
"""

from numpy import *
from scipy.ndimage import filters
from scipy import misc
 
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
    
def cross_bilateral_filter( im, mask_im, d=31, sigma_space=5, sigma_value=0.8 ):
    #filter image im with mask mask_im(commonly the same image im )
    #compute the gaussian Kernel
    
    #resize the mask_im to the same size of im
    if len( mask_im) == 3:
        mask_im = misc.imresize( mask_im, (im.shape[0], im.shape[1], 3) )
    else:
        mask_im = misc.imresize( mask_im, im.shape )
    radius = d / 2 #radius of kernel
    d = radius * 2 + 1
    
    space_weights = createSpaceWeights( sigma_space, d )
    
    
    im_ex = array( boundary_extend( im, radius ), 'f' )
    mask_ex = array( boundary_extend( mask_im, radius ), 'f' )

    
    ( rows, cols ) = im_ex.shape
    if len(mask_im.shape) == 3:
        channels = 3
        value_weights = createValueWeights( sigma_value, 256 * 3 )
    else:
        channels = 1
        value_weights = createValueWeights( sigma_value, 256 )
    
    result = zeros( (rows,cols) )
    if channels == 1:
        for i in arange( radius, rows - radius ):
            for j in arange( radius, cols - radius ):
                imVector = im_ex[i-radius:i+radius+1, j-radius:j+radius+1].flatten()
                
                maskVector = mask_ex[i-radius:i+radius+1, j-radius:j+radius+1].flatten()            
                val0 = mask_ex[i,j]
                
                #!notice:maskBlock has to change to float before minus operate, or uint8(8)-uint8(9) = 255
                valueDistance = array( abs( maskVector - val0 ),'int' )
                
                wVector = space_weights * value_weights[valueDistance]
                
                result[i,j] = sum( imVector * wVector ) / sum(wVector)
                
        return result[ radius:rows-radius, radius:cols-radius ]
    else:
        for i in arange( radius, rows - radius ):
            for j in arange( radius, cols - radius ):
                imVector = im_ex[i-radius:i+radius+1, j-radius:j+radius+1].flatten()
                
                mask3Vector = mask_ex[i-radius:i+radius+1, j-radius:j+radius+1,:].reshape(-1,3).T            
                val0 = mask_ex[i,j,0]
                val1 = mask_ex[i,j,1]
                val2 = mask_ex[i,j,2]
                
                #!notice:maskBlock has to change to float before minus operate, or uint8(8)-uint8(9) = 255
                valueDistance = array( abs( mask3Vector[0,:] - val0 ) + abs( mask3Vector[1,:] - val1 ) + abs( mask3Vector[2,:] - val2 ), 'int' )
                
                wVector = space_weights * value_weights[valueDistance]
                
                result[i,j] = sum( imVector * wVector ) / float(sum(wVector))
                
        return result[ radius:rows-radius, radius:cols-radius ]
        
            

    
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
            r = sqrt( double(i*i) + double(j*j) )
            if r > radius:
                space_weights[ i+ radius, j+radius ] = 0.0
                continue
            space_weights[ i+ radius, j+radius ] = exp( r*r * gaussian_space_coff )
    return space_weights.flatten()

    
def createValueWeights( sigma_value, maxPixelValue ):
    gaussian_value_coff = -0.5/( sigma_value * sigma_value )
    value_weights  = zeros( maxPixelValue )
    for i in range( maxPixelValue ):
        value_weights[i] = exp( i*i * gaussian_value_coff )
    return value_weights
    