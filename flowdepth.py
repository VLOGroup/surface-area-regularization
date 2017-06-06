# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 16:29:58 2015

@author: gottfried
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt4 import QtGui

import flowtools as ft
import warping as warp
import solver as pd

def run_ctf(params, d0, data, imageWidget=None):
    
    levels = (np.floor( (np.log(params['minSize'])-np.log(np.minimum(*d0.shape))) / np.log(params['scalefactor']) ) + 1).astype('int')
    levels = np.minimum(levels, params['levels'])
    warps = params['warps']
    iterations = params['iterations']
    
    L = params['Lambda']          # compute lambda sequence for the levels
    Lambda = np.zeros(levels)
    for idx,l in enumerate(Lambda):
        Lambda[idx] = L*(1.0/params['scalefactor'])**idx
        
    params['Lambda'] = Lambda
    dim_dual = 3        # dimension of dual variable
    
    for level in range(levels-1,params['stop_level']-1,-1):
        level_sz = np.round(np.array(d0.shape) * params['scalefactor']**level)
        factor = np.hstack(( level_sz / np.array(d0.shape), 1 ))
        K = data['K'] * (np.ones((3,3))*factor).T; K[2,2] = 1    # scale K matrix according to image size
                                                               # individual scaling for x & y
        print '--- level %d, size %.1f %.1f' % (level, level_sz[1], level_sz[0])
        
        if level == levels-1:
            d = ft.imresize(d0, level_sz)      # initialization at coarsest level
            p = np.zeros(d.shape + (dim_dual,))
        else:
            print 'prolongate'
            d = ft.imresize(d, level_sz)          # prolongate to finer level
            ptmp = p.copy()
            p = np.zeros(d.shape+(dim_dual,))
            for i in range(0, dim_dual):
                p[:,:,i] = ft.imresize(ptmp[:,:,i], level_sz)
                
        Xn = ft.backproject(d.shape, K)
        L_normal = ft.make_linearOperator(d.shape, Xn, K)
                
        img_scaled = []                    # scale all images
        for img in data['images']:
            img_scaled.append(ft.imresize(img, level_sz))
        
        Iref = img_scaled[0]
        Gref = data['G'][0]
        
            
        for k in range(1,warps+1):
            print 'warp', k, 'maxZ=', params['maxz']
            
            warpdata = []                       # warp all images
            for img,g in zip(img_scaled[1:], data['G'][1:]):
                Grel = ft.relative_transformation(Gref, g)
                warpdata.append(warp.warp_zeta(Iref, img, Grel, K, ft.to_zeta(d)))
                
            d,p,energy = pd.solve_area_pd(warpdata, d, p, iterations, params, params['Lambda'][level], L_normal, imageWidget)
            
    return d



if __name__ == '__main__':
    
    if QtGui.QApplication.instance()==None:
        app=QtGui.QApplication(sys.argv)
        
    plt.close('all')
    
    
    if 'imageWidget' in locals() and imageWidget is not None:
        imageWidget.close()
        imageWidget = None    
        
    npz = np.load('data.npz')
    images = [npz['I1'], npz['I2']]; G = [npz['G1'], npz['G2']]
    data = dict(images=images, G=G, K=npz['K'])
    d_init = 1.8
    
    I1 = data['images'][0]; I2 = data['images'][1]; K = data['K']
    d0 = np.ones(I1.shape)*d_init
        
    #path='/home/gottfried/vmgpu/Playground/stereoGeoRegulariser/data/fountain_dense'
    #data,G,K = OpenEXR.read(os.path.join(path,'frame_'+str(6).zfill(4)+'.exr'))
    #data1,G1,K1 = OpenEXR.read(os.path.join(path,'frame_'+str(7).zfill(4)+'.exr'))
    #images = [data['gray'].astype('f')/255, data1['gray'].astype('f')/255]
    #G = [G, G1]
    #data = dict(images=images, G=G, K=K)
    #I1 = data['images'][0]; I2 = data['images'][1]; K = data['K']
    #d0 = np.ones(I1.shape)*8.3
    
    minalpha = 0.015             # 3d points must be seen at least under this angle
    maxz = ft.camera_baseline(data['G'][0], data['G'][1])/np.arctan(minalpha/2)    # compute depth value
    
    params = dict(warps=15,
                  iterations=30,
                  scalefactor=0.75,
                  levels=100,
                  minSize=48,
                  stop_level=0, 
                  check=10,
                  Lambda=0.004,
                  ref=0,
                  epsilon = 0.001,     # huber epsilon
                  minz=0.5,                 # lower threshold for scene depth
                  maxz=maxz)                # maxz: clamp depth to be smaller than this value

                  
    imageWidget = ft.MplWidget()
    imageWidget.show()
    
    plt.figure('I1'); plt.imshow(I1, cmap='gray'); plt.title('Image 1')
    plt.figure('I2'); plt.imshow(I2, cmap='gray'); plt.title('Image 2')
       
    d = run_ctf(params, d0, data, imageWidget)
    #plt.figure(); plt.imshow(d); plt.colorbar(); plt.title('result')
    
    plt.show()
    
