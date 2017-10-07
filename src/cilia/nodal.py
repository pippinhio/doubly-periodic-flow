#!/usr/bin/env python

from __future__ import division
from math import pi
from fractions import gcd as gcd
import numpy as np
from numpy import sqrt as sqrt
from numpy import sin as sin
from numpy import cos as cos
from numpy import tan as tan
from numpy import pi as pi

from src.evaluation import interp, map_into_main_domain

def parametrize(t,cilia_config,par):
    ncilia_x = cilia_config['ncilia_x']
    ncilia_y = cilia_config['ncilia_y']
    ncilia = ncilia_x*ncilia_y
    npoints = cilia_config['npoints']
    L = cilia_config['L']
    theta = cilia_config['theta']    
    psi = cilia_config['psi']
    T = cilia_config['T']
 
    x_a = par['box']['x_a']
    y_a = par['box']['y_a']
    L_x = par['box']['L_x']
    L_y = par['box']['L_y']

    lcm = ncilia_x*ncilia_y//gcd(ncilia_x,ncilia_y)
    shift_x = np.linspace(0,lcm,ncilia_x,endpoint=False)
    shift_y = np.linspace(0,lcm,ncilia_y,endpoint=False)
    phase_shift = np.mod(shift_x[:,np.newaxis] + shift_y,lcm)

    ds = 1.0/npoints
    s = np.linspace(ds,1.0,npoints,endpoint=True)    
    #Use the following for non-uniform spacing
#    s = np.linspace(ds/2,1.0-ds/2,npoints,endpoint=True)    
#    a = 0.5 # use a = 0 for uniform
#    s = (1-a)*s + a* (0.5 - 0.5*cos(s*pi))

    s = s*L
    r = sin(psi)*s

    X0 = np.zeros((ncilia_x,ncilia_y,npoints),dtype=[('x',float),('y',float),('z',float)])
    U = np.zeros((ncilia_x,ncilia_y,npoints),dtype=[('u',float),('v',float),('w',float)])

    for i in range(ncilia_x):
        for j in range(ncilia_y):
            # CONE
            # First we create a cone whose centerline is the z-axis.
            # Then we rotate the cone about the y-axis by theta.
            x = r*cos(2*pi*(t/T+phase_shift[i,j]/lcm))
            y = r*sin(2*pi*(t/T+phase_shift[i,j]/lcm))
            z = r/tan(psi)
            x_dot = -2*pi/T*r*sin(2*pi*(t/T+phase_shift[i,j]/lcm))
            y_dot =  2*pi/T*r*cos(2*pi*(t/T+phase_shift[i,j]/lcm))
                    
            X0[i,j,:]['x'] = x
            X0[i,j,:]['y'] = y*cos(theta) + z*sin(theta)    
            X0[i,j,:]['z'] = -y*sin(theta) + z*cos(theta)
            
            U[i,j,:]['u'] = x_dot
            U[i,j,:]['v'] = y_dot*cos(theta)    
            U[i,j,:]['w'] = -y_dot*sin(theta)

            # Shift cilium to the correct location.
            X0[i,j,:]['x'] += x_a + (i + 0.5)/ncilia_x*L_x
            X0[i,j,:]['y'] += y_a + (j + 0.5)/ncilia_y*L_y


    # Fit cilium into main domain.
    X0['x'] = map_into_main_domain(X0['x'],par['box'],'x')
    X0['y'] = map_into_main_domain(X0['y'],par['box'],'y')

    X0_vec = np.zeros((ncilia_x*ncilia_y*npoints),dtype=[('x',float),('y',float),('z',float)])
    U_vec = np.zeros((ncilia_x*ncilia_y*npoints),dtype=[('u',float),('v',float),('w',float)])
    X0_vec['x'] = X0['x'].reshape(-1)
    X0_vec['y'] = X0['y'].reshape(-1)
    X0_vec['z'] = X0['z'].reshape(-1)
    U_vec['u'] = U['u'].reshape(-1)
    U_vec['v'] = U['v'].reshape(-1)
    U_vec['w'] = U['w'].reshape(-1)
    
    return (X0_vec,U_vec,phase_shift)



def parametrize_by_hand(t,cilia_config,par):
    base_x      = [ 0.5, 1.5]
    base_y      = [ 0.5, 0.5]
    phase_shift = np.array([ 0.0, 180.0])*pi/180.0

    ncilia = len(base_x)
    npoints = cilia_config['npoints']
    L = cilia_config['L']
    theta = cilia_config['theta']    
    psi = cilia_config['psi']
    T = cilia_config['T']

    s = np.linspace(L,0,npoints,endpoint=False)[::-1]
    r = sin(psi)*s

    X0 = np.zeros((ncilia,npoints),dtype=[('x',float),('y',float),('z',float)])
    U = np.zeros((ncilia,npoints),dtype=[('u',float),('v',float),('w',float)])

    for i in range(ncilia):
        # First we create a cone whose centerline is the z-axis.
        # Then we rotate the cone about the y-axis by theta.
        x = r*cos(2*pi*t/T+phase_shift[i])
        y = r*sin(2*pi*t/T+phase_shift[i])
        z = r/tan(psi)
        x_dot = -2*pi/T*r*sin(2*pi*t/T+phase_shift[i])
        y_dot =  2*pi/T*r*cos(2*pi*t/T+phase_shift[i])
                
        X0[i,:]['x'] = x
        X0[i,:]['y'] = y*cos(theta) + z*sin(theta)    
        X0[i,:]['z'] = -y*sin(theta) + z*cos(theta)
        
        U[i,:]['u'] = x_dot
        U[i,:]['v'] = y_dot*cos(theta)    
        U[i,:]['w'] = -y_dot*sin(theta)

        # Shift cilium to the correct location.
        X0[i,:]['x'] += base_x[i]
        X0[i,:]['y'] += base_y[i]


    # Fit cilium into main domain.
    X0['x'] = map_into_main_domain(X0['x'],par['box'],'x')
    X0['y'] = map_into_main_domain(X0['y'],par['box'],'y')

    X0_vec = np.zeros((ncilia*npoints),dtype=[('x',float),('y',float),('z',float)])
    U_vec = np.zeros((ncilia*npoints),dtype=[('u',float),('v',float),('w',float)])
    X0_vec['x'] = X0['x'].reshape(-1)
    X0_vec['y'] = X0['y'].reshape(-1)
    X0_vec['z'] = X0['z'].reshape(-1)
    U_vec['u'] = U['u'].reshape(-1)
    U_vec['v'] = U['v'].reshape(-1)
    U_vec['w'] = U['w'].reshape(-1)
    
    return (X0_vec,U_vec,phase_shift)

