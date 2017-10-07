#!/usr/bin/env python

from __future__ import division
import os
import time

import numpy as np
from numpy import random as rd

# This is needed to that packages form the parent directory can be
# loaded without using relative imports.
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from core.velocity.velocity_from_forces import velocity_from_forces
from core.set_parameters import set_parameters
from core.data_structures import make_forces_struct

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
F['f1'][-1] = -sum(F['f1'][:-1])
F['f2'][-1] = -sum(F['f2'][:-1])
F['f3'][-1] = -sum(F['f3'][:-1])

dx = 1/64
eps = 4*dx
xi = 6*eps
diff_Ewald_Real = []
diff_Ewald_FFT = []

start = time.time()

par_Real = set_parameters(dx=dx,dy=dx,dz=1/65,method='Real',images=False,epsilon=eps,ncopies_R=40)
forces_Real = make_forces_struct(X0,F,par_Real)
sol_Real = velocity_from_forces(forces_Real,par_Real,cores)
max_Real = max(np.max(sol_Real['u']),np.max(sol_Real['v']),np.max(sol_Real['w']))

par_FFT = set_parameters(dx=dx,dy=dx,dz=1/65,method='FFT',images=False,epsilon=eps)
forces_FFT = make_forces_struct(X0,F,par_FFT)
sol_FFT = velocity_from_forces(forces_FFT,par_FFT,cores)
max_FFT = max(np.max(sol_FFT['u']),np.max(sol_FFT['v']),np.max(sol_FFT['w']))

for i, r_cutoff in enumerate(xi*np.array([1,2,4,6,8])):

    par_Ewald = set_parameters(dx=dx,dy=dx,dz=1/65,method='Ewald',images=False,epsilon=eps,xi=xi,r_cutoff=r_cutoff)
    forces_Ewald = make_forces_struct(X0,F,par_Ewald)
    sol_Ewald = velocity_from_forces(forces_Ewald,par_Ewald,cores)

    r2 = (sol_Real['u']-sol_Ewald['u'])**2 + (sol_Real['v']-sol_Ewald['v'])**2 + (sol_Real['w']-sol_Ewald['w'])**2
    diff_Ewald_Real.append(np.max(np.sqrt(r2)/max_Real))

    r2 = (sol_Ewald['u']-sol_FFT['u'])**2 + (sol_Ewald['v']-sol_FFT['v'])**2 + (sol_Ewald['w']-sol_FFT['w'])**2
    diff_Ewald_FFT.append(np.max(np.sqrt(r2)/max_FFT))
    
    print 'step %g' % (i)
    print 'r_cutoff/dx'
    print r_cutoff/dx
    print 'Real_FFT'
    print diff_Ewald_FFT[-1]
    print 'Real_Ewald'
    print diff_Ewald_Real[-1]


print 'FFT_Real'
print diff_Ewald_Real
print 'Ewald_FFT'
print diff_Ewald_FFT

