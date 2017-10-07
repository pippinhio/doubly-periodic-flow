#!/usr/bin/env python

from __future__ import division
import os

import numpy as np
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

import params

cores = 4

X0 = np.zeros(1,dtype=[('x',float),('y',float),('z',float)])
X0['x'] = 0.5
X0['y'] = 0.5
z0_vec = [0.5,1.0]

F = np.zeros(1,dtype=[('f1',float),('f2',float),('f3',float)])
F['f1'] = 1.0
F['f2'] = 0.0
F['f3'] = 1.0

eps_vec = [10.0**-8,0.0625,0.3]

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.grid(True)

for i in range(len(z0_vec)):
    z0 = z0_vec[i]
    X0['z'] = z0
    
    for j in range(len(eps_vec)):
        eps = eps_vec[j]
        par = set_parameters(domain=[0.0,1.0,0.0,1.0,z0,2.0],dz=1/64,method='FFT',images=True,epsilon=eps)
        forces = make_forces_struct(X0,F,par)
        sol = velocity_from_forces(forces,par,cores)
        z = par['grid']['z_vec']
        L_x = par['box']['L_x']
        L_y = par['box']['L_y']
        u_inf = z0/(L_x*L_y)*F['f1']
        v_inf = z0/(L_x*L_y)*F['f2']
        w_inf = 0.0
        err = np.sqrt( (sol['u'] - u_inf)**2 + (sol['v'] - v_inf)**2 + (sol['w'] - w_inf)**2 )
        err_max = np.squeeze(np.apply_over_axes(np.amax, err, axes=[0,1]))
        
        if i==0: # Add to legend only the first time
            ax1.semilogy(z,err_max,params.linestyles[j], linewidth=params.linewidth,label=r'$\varepsilon = %g$' % eps)
        else:
            ax1.semilogy(z,err_max,params.linestyles[j], linewidth=params.linewidth)

ax1.set_xlim([0.0,2.0])
ax1.set_ylim([10.0**-4,10.0**1])
ax1.set_xlabel(r'$z$',fontsize=params.fontsize_latex)
ax1.set_ylabel(r'$\max_{x,y}( \mathbf{u} - \mathbf{u}_{\infty}) $',fontsize=params.fontsize)
ax1.annotate(r'$z_0=%g$' % z0_vec[0], xy=(1.15, 0.0128), xytext=(0.7, 0.005),
    arrowprops=dict(facecolor='black', shrink=0.05), 
#    horizontalalignment='left',
    fontsize = params.fontsize_latex
    )
ax1.annotate(r'$z_0=%g$' % z0_vec[1], xy=(1.5, 0.07), xytext=(1.7, 0.18),
    arrowprops=dict(facecolor='black', shrink=0.05), 
    fontsize = params.fontsize_latex
    )
ax1.legend(loc=1,prop={'size':params.fontsize_latex})
matplotlib.rcParams.update({'font.size': params.fontsize})
plt.savefig(params.image_path + 'convergence_in_z.eps')
plt.savefig(params.image_path + 'convergence_in_z.pdf')

plt.show()

#import code
#code.interact(local=locals())  

