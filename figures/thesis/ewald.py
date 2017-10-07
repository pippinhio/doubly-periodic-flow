#!/usr/bin/env python

from __future__ import division
import os

import numpy as np
from numpy import pi as pi
import matplotlib
import matplotlib.pyplot as plt

# This is needed to that packages form the parent directory can be
# loaded without using relative imports.
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from core.velocity.velocity_from_forces import velocity_from_forces
from core.set_parameters import set_parameters
from core.data_structures import make_forces_struct
#from core.data_structures import make_forces_struct

import params
# --------------------------------------
# NOTE: 
# In order for this routine to work, in the module velocity_from_forces
# from core.velocity.velocity_from_forces replace the line
#u = u_erf + u_erfc
# with this
#u = par['erf']*u_erf + par['erfc']*u_erfc 
# --------------------------------------


cores = 4

for xi in {0.2,0.3,0.4}:
    par = set_parameters(domain=[-0.0,1.0,-1.0,1.0],dx=1/256,dy=1/64,z_layers=np.array(1.0),method='Ewald',images=False,epsilon=0.04,xi=xi)

    X0 = np.zeros(1,dtype=[('x',float),('y',float),('z',float)])
    X0['x'] = 0.5
    X0['y'] = 0.0
    X0['z'] = 1.0

    F = np.zeros(1,dtype=[('f1',float),('f2',float),('f3',float)])
    F['f1'] = 1.0
    F['f2'] = 0.0
    F['f3'] = 0.0

    forces = make_forces_struct(X0,F,par)

    par.update({'erf':1,'erfc':1})
    sol = velocity_from_forces(forces,par,cores)
    x = par['grid']['x_vec']
    y = par['grid']['y_vec']
    u = np.squeeze(sol['u'][:,y==0,:])

    par.update({'erf':1,'erfc':0})
    sol = velocity_from_forces(forces,par,cores)
    u_erf = np.squeeze(sol['u'][:,y==0,:])

    par.update({'erf':0,'erfc':1})
    sol = velocity_from_forces(forces,par,cores)
    u_erfc = np.squeeze(sol['u'][:,y==0,:])

    fig = plt.figure()
    ax0 = fig.add_subplot(111)
    ax0.plot(x,u     ,params.linestyle1,linewidth=params.linewidth,label='Solution')
    ax0.plot(x,u_erf ,params.linestyle2,linewidth=params.linewidth,label='FFT')
    ax0.plot(x,u_erfc,params.linestyle3,linewidth=params.linewidth,label='Real space')

    ax0.set_xlim([par['box']['x_a'],par['box']['x_b']])
    ax0.legend(prop={'size':params.fontsize})
    ax0.grid(True)

    yticks = ax0.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    matplotlib.rcParams.update({'font.size': params.fontsize})
    ax0.set_xlabel(r'$x$',fontsize=params.fontsize_latex)
    ax0.set_ylabel(r'$u$',fontsize=params.fontsize_latex)
    
    ax0.annotate(r'$\xi=%g$'%xi, xy=(0.02, 2.2), xytext=(0.02, 2.2))
    file_name = ('ewald_%g'%xi).replace('.','_')
    plt.savefig(params.image_path + file_name + '.pdf')
    plt.savefig(params.image_path + file_name + '.eps')
    plt.savefig(params.image_path + file_name + '.png')
    plt.show()


