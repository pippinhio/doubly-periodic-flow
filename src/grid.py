#!/usr/bin/env python

from __future__ import division

import numpy as np
from numpy import pi


def make_grid(box,spacing):
    x_a = box['x_a']
    x_b = box['x_b']
    y_a = box['y_a']
    y_b = box['y_b']
    L_x = box['L_x']
    L_y = box['L_y']
    dx = spacing['dx']
    dy = spacing['dy']
    # The spacing dx_used, dy_used that's handed back with the grid
    # can be slightly different from (that is finer than) the spacing
    # dx, dy that was requested. This happens so that the grid is
    # uniform over the domain of size L_x * L_y.
    N_x = int(np.ceil(L_x/dx))
    N_y = int(np.ceil(L_y/dy))
    (x,dx_used) = np.linspace(x_a,x_b,num=N_x,endpoint=False,retstep=True)
    (y,dy_used) = np.linspace(y_a,y_b,num=N_y,endpoint=False,retstep=True)

    # compute wave numbers
    k = 2*pi*N_x/L_x*np.fft.fftfreq(N_x)
    m = 2*pi*N_y/L_y*np.fft.fftfreq(N_y)

    uniform_in_z = spacing['uniform_in_z']
    if uniform_in_z:
        z_a = box['z_a']
        z_b = box['z_b']
        L_z = box['L_z']
        dz = spacing['dz']
        N_z = int(np.ceil(L_z/dz)) + 1
        (z,dz_used) = np.linspace(z_a,z_b,num=N_z,endpoint=True,retstep=True)
        # If there is only one z_layer the grid is classified as
        # nonuniform.
    else:
        # Note: numpy.unique gets rid of double elements
        # and also sorts z_layers.
        z_layers = spacing['z_layers']
        z = np.unique(z_layers)
        N_z = len(z)

    (xx,yy,zz) = ndgrid(x,y,z)
    (kk,mm) = ndgrid(k,m,z)[0:2]

    grid = {'x':xx,'y':yy,'z':zz,'k':kk,'m':mm,
            'N_x':N_x,'N_y':N_y,'N_z':N_z,
            'dx':dx,'dy':dy,'uniform_in_z':uniform_in_z,
            'shape':(N_x,N_y,N_z),
            'x_vec':x,'y_vec':y,'z_vec':z
           }
    if uniform_in_z:
        grid.update({'dz':dz_used})
    else:
        grid.update({'z_layers':z_layers})

    return grid



def ndgrid(*args):
    # This function will become obsolete in numpy 1.8 when numpy.meshgrid
    # is able to handle arrays of arbitray dimension.
    if len(args) == 2:
        x = args[0]
        z = args[1]
        x = x[:,np.newaxis]
        z = z[:]
        xx =   x + 0*z
        zz = 0*x +   z
        return (xx,zz)
    elif len(args) == 3:
        x = args[0]
        y = args[1]
        z = args[2]
        x = x[:,np.newaxis,np.newaxis]
        y = y[:,np.newaxis]
        z = z[:]
        xx =   x + 0*y + 0*z
        yy = 0*x +   y + 0*z
        zz = 0*x + 0*y +   z
        return (xx,yy,zz)
