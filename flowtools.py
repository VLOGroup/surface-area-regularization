# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 16:31:45 2015

@author: gottfried
"""

import numpy as np
from PyQt4 import QtGui
import scipy.sparse as sparse
from scipy import ndimage


from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class MplWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(MplWidget, self).__init__(parent)

        self.figure = Figure()
        
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.cb = None
        self.im = None
        self.imsz = None
        
    def imshow(self, img):
        if self.im:
            self.imsz = self.im.get_size()
            newsz = img.shape
            self.im.set_data(img)
            if self.imsz[0] != newsz[0] or self.imsz[1] != newsz[1]:    # update extent
                self.im.set_extent((-0.5, newsz[1]-0.5, newsz[0]-0.5, -0.5))            
        else:
            self.im = self.ax.imshow(img,interpolation='none')
        
        if self.cb:
            self.im.autoscale()
        else:
            self.cb = self.figure.colorbar(self.im)
            
        report_pixel = lambda x, y : "(%6.3f, %6.3f) %.3f" % (x,y, img[np.floor(y+0.5),np.floor(x+0.5)])
        self.ax.format_coord = report_pixel
            
        self.canvas.draw()
        self.canvas.flush_events()


def imresize(img, sz):
    """
    Resize image
    
    Input:
        img: A grayscale image
        sz: A tuple with the new size (rows, cols)
        
    Output:
        Ir: Resized image
    """
    if np.all(img.shape==sz):
        return img;
        
    factors = (np.array(sz)).astype('f') / (np.array(img.shape)).astype('f')

    if np.any(factors < 1):               # smooth before downsampling
        sigmas = (1.0/factors)/3.0
        #print img.shape, sz, sigmas
        I_filter = ndimage.filters.gaussian_filter(img,sigmas)
    else:
        I_filter = img
        
    u,v = np.meshgrid(np.arange(0,sz[1]).astype('f'), np.arange(0,sz[0]).astype('f'))
    fx = (float(img.shape[1])) / (sz[1])     # multiplicative factors mapping new coords -> old coords
    fy = (float(img.shape[0])) / (sz[0])
        
    u *= fx; u += (1.0/factors[1])/2 - 1 + 0.5    # sample from correct position
    v *= fy; v += (1.0/factors[0])/2 - 1 + 0.5
    
    # bilinear interpolation
    Ir = ndimage.map_coordinates(I_filter, np.vstack((v.flatten().transpose(),u.flatten().transpose())), order=1, mode='nearest')
    Ir = np.reshape(Ir, (sz[0], sz[1]))
    return Ir


def backproject(shape, K):
    x,y = np.meshgrid(np.arange(0, shape[1]), np.arange(0, shape[0]))
        
    fx = K[0,0]; fy = K[1,1]; cx = K[0,2]; cy = K[1,2]
    Xn = np.ones((3, x.size))
    
    Xn[0,:] = (x.flatten() - cx) / fx
    Xn[1,:] = (y.flatten() - cy) / fy
    
    return Xn



def relative_transformation(G0, G1):
    G01 = np.zeros((3,4))
    G01[0:3,0:3] = np.dot(G1[0:3,0:3], G0[0:3,0:3].transpose()) 
    G01[0:3,3] = G1[0:3,3] - np.dot(G01[0:3,0:3], G0[0:3,3]) 
    return G01
    

        
    
def make_derivatives_2D_complete(shape):
    r"""
    Sparse matrix approximation of gradient operator on image plane.
    Use forward differences inside image, backward differences at left/bottom border
    
    :param shape: image size (tuple of ints)
    
    Returns: 
    
    :Kx,Ky: sparse matrices for gradient in x- and y-direction
    """
    M = shape[0]
    N = shape[1]
    
    x,y = np.meshgrid(np.arange(0,N), np.arange(0,M))
    linIdx = np.ravel_multi_index((y,x), x.shape)    # linIdx[y,x] = linear index of (x,y) in an array of size MxN

    i = np.vstack( (np.reshape(linIdx[:,:-1], (-1,1) ), np.reshape(linIdx[:,:-1], (-1,1) )) )  # row indices
    j = np.vstack( (np.reshape(linIdx[:,:-1], (-1,1) ), np.reshape(linIdx[:,1:], (-1,1) )) )   # column indices
    v = np.vstack( (np.ones( (M*(N-1),1) )*-1, np.ones( (M*(N-1),1) )) )                       # values
    i = np.vstack( (i, np.reshape(linIdx[:,-1], (-1,1) ), np.reshape(linIdx[:,-1], (-1,1) )) )  # row indices
    j = np.vstack( (j, np.reshape(linIdx[:,-1], (-1,1) ), np.reshape(linIdx[:,-2], (-1,1) )) )   # column indices
    v = np.vstack( (v, np.ones( ((M),1) )*1, np.ones( ((M),1) )*-1) )                       # values
    Kx = sparse.coo_matrix((v.flatten(),(i.flatten(),j.flatten())), shape=(M*N,M*N))

    i = np.vstack( (np.reshape(linIdx[:-1,:], (-1,1) ), np.reshape(linIdx[:-1,:], (-1,1) )) )
    j = np.vstack( (np.reshape(linIdx[:-1,:], (-1,1) ), np.reshape(linIdx[1:,:], (-1,1) )) )
    v = np.vstack( (np.ones( ((M-1)*N,1) )*-1, np.ones( ((M-1)*N,1) )) )
    i = np.vstack( (i, np.reshape(linIdx[-1,:], (-1,1) ), np.reshape(linIdx[-1,:], (-1,1) )) )
    j = np.vstack( (j, np.reshape(linIdx[-1,:], (-1,1) ), np.reshape(linIdx[-2,:], (-1,1) )) )
    v = np.vstack( (v, np.ones( ((N),1) )*1, np.ones( ((N),1) )*-1) )
    Ky = sparse.coo_matrix((v.flatten(),(i.flatten(),j.flatten())), shape=(M*N,M*N))

    return Kx.tocsr(),Ky.tocsr()    

    
def make_linearOperator(shape, Xn, K):
    
    M,N = shape
    
    fx = K[0,0]
    fy = K[1,1]
    
    x_hat = Xn[0,:]
    y_hat = Xn[1,:]
    
    Kx,Ky = make_derivatives_2D_complete(shape)       # use one-sided differences with backward diff at image border
    Kx = Kx.tocsr()
    Ky = Ky.tocsr()
    
    spId = sparse.eye(M*N, M*N, format='csr')
    spXhat = sparse.diags(x_hat.flatten(), 0).tocsr()
    spYhat = sparse.diags(y_hat.flatten(), 0).tocsr()
    
    L = sparse.vstack([-Kx/fy, -Ky/fx, 
                       spXhat*Kx/fy + spYhat*Ky/fx +
                       2*spId/(fx*fy)
                        ])
    
    return L.tocsr()
    
    

def to_zeta(z):
    return (z**2) / 2
    
def to_z(zeta):
    return np.sqrt(2*zeta)


    
def huber_function(data, epsilon):
    
    idx1 = np.where(np.abs(data)>=epsilon); idx2 = np.where(np.abs(data)<epsilon)
    result = np.zeros(data.shape)
    result[idx1] = np.abs(data[idx1])-0.5*epsilon
    result[idx2] = np.abs(data[idx2])**2/(2*epsilon)
    
    return result
    
def camera_baseline(G0, G1):
    p0 = np.dot(-G0[0:3,0:3].T, G0[0:3,3])
    p1 = np.dot(-G1[0:3,0:3].T, G1[0:3,3])
    
    return np.linalg.norm(p0-p1)