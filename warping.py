# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 16:34:56 2015

@author: gottfried
"""

import numpy as np
from scipy import ndimage

from flowtools import to_z
from flowtools import backproject

    
def warp_zeta(Iref, I, Grel, K, zeta):
    
    Xn = backproject(zeta.shape, K)
    X = np.vstack((Xn, 1/to_z(zeta.flatten())))    # homogeneous coordinate = inverse depth
    
    if Grel.shape[0] < 4:
        Grel = np.vstack((Grel, np.array([0,0,0,1])))
        
    dX = np.dot(Grel[0:3,0:3], Xn)*1/to_z(zeta.flatten())   # derivative of transformation G*X
    
    X2 = np.dot(Grel, X)             # transform point
    x2 = np.dot(K, X2[0:3,:])        # project to image plane
    
    x2[0,:] /= x2[2,:]             # dehomogenize
    x2[1,:] /= x2[2,:]
    Iw = ndimage.map_coordinates(I, np.vstack(np.flipud(x2[0:2,:])), order=1, cval=np.nan)
    Iw = np.reshape(Iw, I.shape)      # warped image
    gIwy,gIwx = np.gradient(Iw)
    
    fx = K[0,0]; fy = K[1,1]
    for i in range(0,3):           # dehomogenize
        X2[i,:] /= X2[3,:]
    z2 = X2[2,:]**2
    dT = np.zeros((2,X2.shape[1]))
    dT[0,:] = fx/X2[2,:]*dX[0,:] - fx*X2[0,:]/z2*dX[2,:]    # derivative of projected point x2 = pi(G*X)
    dT[1,:] = fy/X2[2,:]*dX[1,:] - fy*X2[1,:]/z2*dX[2,:]
    
    # full derivative I(T(x,z)), T(x,z)=pi(G*X)
    Ig = np.reshape(gIwx.flatten()*dT[0,:] + gIwy.flatten()*dT[1,:], zeta.shape)
    It = Iw - Iref       # 'time' derivative'
    
    It[np.where(np.isnan(Iw))] = 0;
    Ig[np.where( (np.isnan(gIwx).astype('uint8') + np.isnan(gIwy).astype('uint8'))>0 ) ] = 0
    
    return Iw, It, Ig