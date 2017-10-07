#!/usr/bin/env python

from __future__ import division

import numpy as np
import warnings

from scipy.ndimage.interpolation import map_coordinates

def interp(U,Xp,par,dim=3):
    # The function U is defined on a grid that has the form
    # (x_a,...,x_b-dx) x (y_a,...,y_b-dy) x (z_a,...,z_b).
    # At first, the "missing" boundaries at x_b and y_b are added.
    # (Recall that U is periodic in x and y).
    if dim == 2:
        U = np.append(U,U[0,:][np.newaxis,:],0)
    if dim == 3:
        U = np.append(U,U[0,:,:][np.newaxis,:,:],0)
        U = np.append(U,U[:,0,:][:,np.newaxis,:],1)

    # The are no limits on the interpolation points in the x and y-direction
    # since the function U is assumed to be periodic in x and y.  In the
    # z-direction points have to be within [z_a,z_b].
    xp = map_into_main_domain(Xp['x'],par['box'],'x')
    if dim == 3:
        yp = map_into_main_domain(Xp['y'],par['box'],'y')
    zp = Xp['z']
    z_a = par['box']['z_a']
    z_b = par['box']['z_b']
    if (zp < z_a).any() or (zp > z_b).any():
        warnings.warn('Tried to intepolate points with z-coordinate '
                      + 'outside of domain.')

    # The function map_coordinates uses indices rather than values.
    # For instance, the point x_a + 2*dx has index idx = 2.
    x_a = par['box']['x_a']
    dx = par['grid']['dx']
    xp_idx = (xp - x_a)/dx
    if dim == 3:
        y_a = par['box']['y_a']
        dy = par['grid']['dy']
        yp_idx = (yp - y_a)/dy

    uniform_in_z = par['grid']['uniform_in_z']
    if uniform_in_z:
        dz = par['grid']['dz']
        zp_idx = (zp - z_a)/dz
    else:
        # If the grid is nonunifrom in z, the function cannot be
        # interpolated in z but merely evauated at the given z layers.
        z_layers = par['grid']['z_layers']
        zp_idx = zp.copy()
        zp_idx.fill(None)
        for i in range(len(z_layers)):
            zp_idx[zp==z_layers[i]] = i
        if np.isnan(zp_idx).any():
            warnings.warn('Cannot interpolate grid that is not uniform in z.')
    if dim == 2:
        Up = map_coordinates(U,[xp_idx,zp_idx],order=3)
    elif dim == 3:
        Up = map_coordinates(U,[xp_idx,yp_idx,zp_idx],order=3)
    return Up


def map_into_main_domain(vec,box,dim):
    if dim == 'x':
        vec = (vec - box['x_a']) % box['L_x'] + box['x_a']
    elif dim == 'y':
        vec = (vec - box['y_a']) % box['L_y'] + box['y_a']
    return vec

