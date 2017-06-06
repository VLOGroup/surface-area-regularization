# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 09:44:06 2015

@author: gottfried
"""

import numpy as np

from flowtools import huber_function
from flowtools import to_z
from flowtools import to_zeta


def solve_area_pd(warpdata, d0, p, iterations, params, Lambda, L, imagewidget=None):
    numviews = len(warpdata)
    
    check = params['check']
    epsilon = params['epsilon']
    
    Iw = warpdata[0][0]
    It = warpdata[0][1]
    Ig = warpdata[0][2]
    
    Lambda /= numviews
    print 'Lambda=', Lambda, 'numviews=', numviews

    
    zeta0 = to_zeta(d0)
    zeta = zeta0.copy()
    zeta_ = zeta0.copy()
    energy = []        
        
    tau = np.zeros(L.shape[1])           # compute preconditioners
    sigma = np.zeros(L.shape[0])
    tau = 1/np.array(np.abs(L).sum(axis=0)).flatten()
    sigma = 1/np.array(np.abs(L).sum(axis=1)).flatten()
    tau[np.where(np.isinf(tau))] = 0
    sigma[np.where(np.isinf(sigma))] = 0
    

    for k in range(1,iterations+1):
        
        p = p.flatten()
        p += sigma*(L*zeta_.flatten())    # dual update
        p = np.reshape(p, (3,-1))
        
        
        normp = np.sqrt(p[0,:]**2 + p[1,:]**2 + p[2,:]**2)
        reprojection = np.maximum(1, normp)
        p[0,:] /= reprojection
        p[1,:] /= reprojection
        p[2,:] /= reprojection
        p = p.flatten()
        
        zeta_ = zeta.copy()      # remember zeta
        zeta = zeta.flatten()
        
        zeta -= tau*(L.T*p)       # primal update
        zeta = np.reshape(zeta, zeta0.shape)
        tau = np.reshape(tau, zeta0.shape)    # prox
        r = It + Ig*(zeta-zeta0)
        th = epsilon + tau*Lambda*Ig**2
        idx1 = np.where(r > th)
        idx2 = np.where(r < -th)
        idx3 = np.where(np.abs(r) <= th)
        zeta[idx1] -= tau[idx1]*Lambda*Ig[idx1]
        zeta[idx2] += tau[idx2]*Lambda*Ig[idx2]
        zeta[idx3] = (zeta[idx3] - tau[idx3]*Lambda*Ig[idx3]*(It[idx3]-zeta0[idx3]*Ig[idx3]) / epsilon) / (1+tau[idx3]*Lambda*Ig[idx3]**2/epsilon)
        tau = tau.flatten()
        
        zeta = np.maximum(to_zeta(params['minz']), zeta)        # make sure depth stays positive
        zeta = np.minimum(to_zeta(params['maxz']), zeta)          # clamp
        zeta_ = 2*zeta-zeta_                    # extra gradient step
        
        grad = L*zeta.flatten()              # compute energy
        grad = np.reshape(grad, (3,-1))
        normgrad = np.sqrt(grad[0,:]**2 + grad[1,:]**2 + grad[2,:]**2)
        r = huber_function(It+(zeta-zeta0)*Ig, epsilon)
        energy.append(normgrad.sum() + Lambda*r.sum())        
                
        if k % check == 0:
            print 'iter ', k, ' energy ', energy[-1]
            if imagewidget is not None:
                imagewidget.imshow(to_z(zeta))
                #imagewidget.imshow(r)      # show the residual
                
            
        p1 = np.zeros(zeta0.shape+(3,))
        p = np.reshape(p, (-1,3))
        p1[:,:,0] = np.reshape(p[:,0], zeta0.shape)
        p1[:,:,1] = np.reshape(p[:,1], zeta0.shape)
        p1[:,:,2] = np.reshape(p[:,2], zeta0.shape)
        p=p1
    return to_z(zeta), p, energy
