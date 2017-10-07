#!/usr/bin/env python

from __future__ import division
import os

import numpy as np
from numpy import random as rd
import matplotlib
import matplotlib.pyplot as plt

# This is needed to that packages form the parent directory can be
# loaded without using relative imports.
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from src.velocity.velocity_from_forces import velocity_from_forces
from src.set_parameters import set_parameters
from src.data_structures import make_forces_struct

import params

cores = 4

nforces = 10

rd.seed(1)
X0 = np.zeros(nforces,dtype=[('x',float),('y',float),('z',float)])
X0['x'] = rd.rand(nforces)
X0['y'] = rd.rand(nforces)
X0['z'] = rd.rand(nforces)

F = np.zeros(nforces,dtype=[('f1',float),('f2',float),('f3',float)])
F['f1'] = rd.rand(nforces)
F['f2'] = rd.rand(nforces)
F['f3'] = rd.rand(nforces)

# make sure the netforce is equal to zero
#F['f1'][-1] = -sum(F['f1'][:-1])
#F['f2'][-1] = -sum(F['f2'][:-1])
#F['f3'][-1] = -sum(F['f3'][:-1])

dx = 1/64

N = 16
epsilons = np.linspace(5/N,5,N,endpoint=True)*dx
diff = np.zeros(epsilons.shape)
i = 0
for eps in epsilons:
    par_Ewald = set_parameters(dx=dx,dy=dx,dz=1/65,method='Ewald',images=False,epsilon=eps,xi=4*dx)
    forces_Ewald = make_forces_struct(X0,F,par_Ewald)
    sol_Ewald = velocity_from_forces(forces_Ewald,par_Ewald,cores)

    par_FFT = set_parameters(dx=dx,dy=dx,dz=1/65,method='FFT',images=False,epsilon=eps)
    forces_FFT = make_forces_struct(X0,F,par_FFT)
    sol_FFT = velocity_from_forces(forces_FFT,par_FFT,cores)

    r2 = (sol_Ewald['u']-sol_FFT['u'])**2 + (sol_Ewald['v']-sol_FFT['v'])**2 + (sol_Ewald['w']-sol_FFT['w'])**2
    max_Ewald = max(np.max(sol_Ewald['u']),np.max(sol_Ewald['v']),np.max(sol_Ewald['w']))

    diff[i] = np.max(np.sqrt(r2)/max_Ewald)
    i += 1
    print 'step %g from %g' % ((i),len(epsilons))

fig = plt.figure()
ax0 = fig.add_subplot(111)
ax0.semilogy(epsilons/dx,diff,params.linestyle1,marker='s',label='Ewald vs. FFT')
ax0.grid(True)
ax0.set_xlim([0.0,6.0])
ax0.set_ylim([10.0**-16,10.0])

ax0.legend(loc=1)

ax0.set_xlabel(r'$\varepsilon / \Delta x$',fontsize=params.fontsize_latex)
ax0.set_ylabel('relative error in max norm',fontsize=params.fontsize)
#plt.gcf().subplots_adjust(bottom=0.15) #make room for x-label
matplotlib.rcParams.update({'font.size': params.fontsize})
plt.savefig(params.image_path + 'FFT_Ewald.pdf')
plt.savefig(params.image_path + 'FFT_Ewald.eps')
plt.show()


