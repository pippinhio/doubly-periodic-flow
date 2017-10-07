#!/usr/bin/env python

from __future__ import division
#import time
import os
#import pickle

import numpy as np
#from numpy import pi as pi
import matplotlib
import matplotlib.pyplot as plt

# This is needed to that packages form the parent directory can be
# loaded without using relative imports.
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
#from src.velocity.forces_from_velocity import forces_from_velocity
from src.velocity.velocity_from_forces import velocity_from_forces
from src.set_parameters import set_parameters
from src.data_structures import make_forces_struct
import params

cores = 4

for case in {'free_no-img','free_img','dp_no-img','dp_img'}:
    if case == 'free_no-img':
        method = 'Real'
        images = False
    elif case == 'free_img':
        method = 'Real'
        images = True
    elif case == 'dp_no-img':
        method = 'FFT'
        images = False
    elif case == 'dp_img':
        method = 'FFT'
        images = True

    par = set_parameters(domain=[-1.0,1.0,-1.0,1.0,0.0,2.0],dx=1/64,dy=1/64,dz=1/11,method=method,images=images,epsilon=0.2)

    X0 = np.zeros(1,dtype=[('x',float),('y',float),('z',float)])
    X0['x'] = 0.0
    X0['y'] = 0.0
    X0['z'] = 1.0

    F = np.zeros(1,dtype=[('f1',float),('f2',float),('f3',float)])
    F['f1'] = 1.0
    F['f2'] = 0.0
    F['f3'] = 1.0

    forces = make_forces_struct(X0,F,par)
    sol = velocity_from_forces(forces,par,cores)

    y = par['grid']['y_vec']
    xx = np.squeeze(par['grid']['x'][:,y==0,:]) 
    zz = np.squeeze(par['grid']['z'][:,y==0,:]) 
    u = np.squeeze(sol['u'][:,y==0,:])
    w = np.squeeze(sol['w'][:,y==0,:])

    fig = plt.figure()
    ax0 = fig.add_subplot(111)
    ax0.quiver(xx[::4],zz[::4],u[::4],w[::4])
    ax0.quiver(X0['x'],X0['z'],F['f1'],F['f3'],color='red',scale=10,width=0.02)
    #ax0.axhline(y=0.0,linestyle='dashed',color='black')
    ax0.set_ylim([-0.1,2.0])
    ax0.set_xlabel(r'$x$',fontsize=params.fontsize_latex)
    ax0.set_ylabel(r'$z$',fontsize=params.fontsize_latex)

    if case == 'free_no-img':
            ax0.set_title('Stokeslet in free space')   
    elif case == 'free_img':
            ax0.set_title('Stokeslet near a wall')   
    elif case == 'dp_no-img':
            ax0.set_title('Doubly-periodic Stokeslet in free space')   
    elif case == 'dp_img':
            ax0.set_title('Doubly-periodic Stokeslet near a wall')   
    matplotlib.rcParams.update({'font.size': params.fontsize})
            
    plt.savefig(params.image_path + 'stokeslet_%s.eps' % case)
    plt.savefig(params.image_path + 'stokeslet_%s.pdf' % case)

    plt.show()

