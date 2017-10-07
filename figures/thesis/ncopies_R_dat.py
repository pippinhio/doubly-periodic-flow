#!/usr/bin/env python

from __future__ import division
import os

import numpy as np
from numpy import random as rd

# This is needed to that packages form the parent directory can be
# loaded without using relative imports.
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from src.velocity.velocity_from_forces import velocity_from_forces
from src.set_parameters import set_parameters
from src.data_structures import make_forces_struct

# In order to compute the reference solution without adjusting the plane, 
# disable the following lines in velocity_from_forces_serial in
# src.velocity.velocity_from_forces
#    u += plane_u
#    v += plane_v
#    w += plane_w

cores = 4

nforces = 4

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

par_ref = set_parameters(dx=dx,dy=dx,dz=1/11,method='FFT',images=False,epsilon=eps)
forces_ref = make_forces_struct(X0,F,par_ref)
sol_ref = velocity_from_forces(forces_ref,par_ref,cores)

vals = np.array([1,2,4,7,10,20,40,80,160,320])
diff = np.zeros(vals.shape)
i = 0
for ncopies_R in vals:
    par = set_parameters(dx=dx,dy=dx,dz=1/11,method='Real',images=False,epsilon=eps,ncopies_R=ncopies_R)
    forces = make_forces_struct(X0,F,par)
    sol = velocity_from_forces(forces,par,cores)

    r2 = (sol['u']-sol_ref['u'])**2 + (sol['v']-sol_ref['v'])**2 + (sol['w']-sol_ref['w'])**2
    sol_max = max(np.max(sol['u']),np.max(sol['v']),np.max(sol['w']))
    diff[i] = np.max(np.sqrt(r2)/sol_max)
    i += 1
    print 'step %g from %g' % ((i),len(vals))
    

print vals
print diff


