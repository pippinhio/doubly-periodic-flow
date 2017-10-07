#!/usr/bin/env python

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

# This is needed to that packages form the parent directory can be
# loaded without using relative imports.
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from src.set_parameters import set_parameters
from test.help_routines import run

X0 = np.zeros(2,dtype=[('x',float),('y',float),('z',float)])
X0['x'] = [0.2, 0.8]
X0['y'] = [0.0, 0.0]
X0['z'] = [0.4, 0.8]

F = np.zeros(2,dtype=[('f1',float),('f2',float),('f3',float)])
F['f1'] = [1.0,-1.0]
F['f2'] = [0.0, 0.0]
F['f3'] = [0.5,-0.5]

#For more details about the input parameters see also the docstring in the 
#module src.set_parameters.  Commenting out a line will use the default value. 
#In particualar, set_parameters can be called without any input values.
par = set_parameters( 
    domain=[0.0,1.0,0.0,1.0,0.0,1.0],#compuational domain: [x_a,x_b,y_a,y_b,z_a,z_b]
    dx=1/64,dy=1/64,dz=1/64,#grid spacing
#    z_layers=None,#use for non-uniform grid in z
    images=True,#turn image system on and off
    epsilon=1/64,#regularization parameter
    method='Real',#choices are 'Real', 'FFT', 'Ewald'
#    xi=4/64,#splitting parameter 
#    r_cutoff=None,#cutoff radius for local piece in Ewald splitting
    ncopies_R=0 #number of terms in Stokeslet-double sum (per dimension)
    )

U = run(X0, F, par, cores=1)#set cores = 1 to run code in serial

fig = plt.figure()
ax = fig.add_subplot(111)
xx = par['grid']['x']
yy = par['grid']['y']
zz = par['grid']['z']
ax.quiver(
    xx[yy==0],#This works only if y=0 lies on the grid. Otherwise use interp
    zz[yy==0],#from src.evaluation
    U['u'][yy==0],
    U['w'][yy==0])
ax.set_xlabel(r'$x$',fontsize=18)
ax.set_ylabel(r'$z$',fontsize=18)
plt.show()
