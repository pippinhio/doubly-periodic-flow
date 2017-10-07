#!/usr/bin/env python

from __future__ import division
from fractions import gcd as gcd

import numpy as np
from numpy import sqrt as sqrt
from numpy import sin as sin
from numpy import cos as cos
from numpy import tan as tan
from numpy import pi as pi

from src.evaluation import interp, map_into_main_domain


def parametrize(t,cilia_config,par):

    x_a = par['box']['x_a']
    y_a = par['box']['y_a']
    L_x = par['box']['L_x']
    L_y = par['box']['L_y']

    npoints = cilia_config['npoints']
    s = np.linspace(1,0,npoints,endpoint=False)
    
    ncilia = cilia_config['ncilia']
    X0 = np.zeros(npoints,dtype=[('x',float),('y',float),('z',float)])
    U = np.zeros(npoints,dtype=[('u',float),('v',float),('w',float)])
    
    X0 = np.tile(X0,(ncilia,1))
    U = np.tile(U,(ncilia,1))

    beat_pattern = cilia_config['beat_pattern']
    (a0_x,a_x,b_x,a0_z,a_z,b_z,n) = set_fourier_coefficients(beat_pattern,s)

    T = cilia_config['T']
    for j in range(ncilia):
        # Initially the cilium is attached to the wall at the origin (0,0,0).
        sigma = 2*pi
        sine   = sin(n*sigma*(t/T + j/ncilia))
        cosine = cos(n*sigma*(t/T + j/ncilia))

        X0[j]['x'] = 0.5*a0_x + a_x.dot(cosine) + b_x.dot(sine)
        X0[j]['y'] = 0.0
        X0[j]['z'] = 0.5*a0_z + a_z.dot(cosine) + b_z.dot(sine)

        U[j]['u'] = -a_x.dot(sigma*n*sine) + b_x.dot(sigma*n*cosine)
        U[j]['v'] = 0.0
        U[j]['w'] = -a_z.dot(sigma*n*sine) + b_z.dot(sigma*n*cosine)

        # Shift cilium to the correct location.
        X0[j]['x'] += x_a + (j + 0.5)/ncilia*L_x
        X0[j]['y'] += y_a + 0.5*L_y

        # Fit cilium into main domain.
        X0[j]['x'] = (X0[j]['x'] - x_a) % L_x + x_a
#    import code
#    code.interact(local=locals())  
    X0 = X0.reshape(ncilia*npoints)
    U = U.reshape(ncilia*npoints)
    
    return (X0,U)


def set_fourier_coefficients(beat_pattern,s):
    
    if beat_pattern == 'Sleigh':
        # Fourier least squares coefficinets from Sleigh (1977): 
        # The  nature and action of respiratory tract cilia. Respiratory 
        # Defence Mechanisms. pp. 247-288.
        A_x = np.array([
            [-0.654, 0.787, 0.202],
            [ 0.393,-1.516, 0.716],
            [-0.097, 0.032,-0.118],
            [ 0.079,-0.302, 0.142],
            [ 0.119,-0.252, 0.110],
            [ 0.119,-0.015,-0.013],
            [ 0.009, 0.035,-0.043]
        ])
        A_z = np.array([
            [ 1.895,-0.552, 0.096],
            [-0.018,-0.126, 0.263],
            [ 0.158,-0.341, 0.186],
            [ 0.010, 0.035,-0.067],
            [ 0.003, 0.006,-0.032],
            [ 0.013,-0.029,-0.002],
            [ 0.040,-0.068, 0.015]
        ])
        B_x = np.array([
            [ 0.284, 1.045,-1.017],
            [ 0.006, 0.317,-0.276],
            [-0.059, 0.226,-0.196],
            [ 0.018, 0.004,-0.037],
            [ 0.053,-0.082, 0.025],
            [ 0.009,-0.040,-0.023]
        ])
        B_z = np.array([
            [ 0.192,-0.499, 0.339],
            [-0.050, 0.423,-0.327],
            [ 0.012, 0.138,-0.114],
            [-0.007, 0.125,-0.105],
            [-0.014, 0.075,-0.057],
            [-0.017, 0.067,-0.055]
        ])
    elif beat_pattern == 'SandersonSleigh':
        # Fourier least squares coefficinets from  Sanderson and Sleigh (1981): 
        # Ciliary activity and cultured rabbit epithelium: beat pattern and 
        # mechatrony. J.Cell Sci. 47, 331. 
        A_x = np.array([
            [-0.449,-0.072, 0.658],
            [ 0.130,-1.502, 0.793],
            [-0.169, 0.260,-0.251],
            [ 0.063,-0.123, 0.049],
            [-0.050, 0.011, 0.009],
            [-0.040,-0.009, 0.023],
            [-0.068, 0.196,-0.111]
        ])
        A_z = np.array([
            [ 2.076,-1.074, 0.381],
            [-0.003,-0.230, 0.331],
            [ 0.054,-0.305, 0.193],
            [ 0.007,-0.180, 0.082],
            [ 0.026,-0.069, 0.029],
            [ 0.022, 0.001, 0.002],
            [ 0.010,-0.080, 0.048]
        ])
        B_x = np.array([
            [-0.030, 1.258,-1.034],
            [-0.093,-0.036, 0.050],
            [ 0.037,-0.244, 0.143],
            [ 0.062,-0.093, 0.043],
            [ 0.016,-0.137, 0.098],
            [-0.065, 0.095,-0.054]
        ])
        B_z = np.array([ 
            [ 0.080,-0.298, 0.210],
            [-0.044, 0.513,-0.367],
            [-0.017, 0.004, 0.009],
            [ 0.052,-0.222, 0.120],
            [ 0.007, 0.035,-0.024],
            [ 0.051,-0.128, 0.102] 
        ])

    sp = np.array([s,s**2,s**3])

    a0_x = np.dot(A_x[0,:],sp).T
    a_x  = np.dot(A_x[1:,:],sp).T
    b_x  = np.dot(B_x,sp).T

    a0_z = np.dot(A_z[0,:],sp).T
    a_z = np.dot(A_z[1:,:],sp).T
    b_z = np.dot(B_z,sp).T

    # The number of frequencies N0 corresponds to the size of the above 
    # matrices and is therefore not trivial to change.
    N0 = 6 
    n = np.linspace(1,N0,N0)

    return (a0_x,a_x,b_x,a0_z,a_z,b_z,n)
    
