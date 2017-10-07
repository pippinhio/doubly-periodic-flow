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

import params

cores = 4

nforces = 4

rd.seed(1)
X0 = np.zeros(nforces,dtype=[('x',float),('y',float),('z',float)])
X0['x'] = rd.rand(nforces)
X0['y'] = rd.rand(nforces)
z0_vec = [0.5,0.1,0.05]
eps_vec = [0.0625,0.3]

F = np.zeros(nforces,dtype=[('f1',float),('f2',float),('f3',float)])
F['f1'] = rd.rand(nforces)
F['f2'] = rd.rand(nforces)
F['f3'] = rd.rand(nforces)

for z0 in z0_vec:
    X0['z'] = z0*np.ones(nforces)
    for eps in eps_vec:
        par = set_parameters(z_layers=np.array([0]),method='FFT',images=True,epsilon=eps)
        forces = make_forces_struct(X0,F,par)
        sol = velocity_from_forces(forces,par,cores)
        diff = np.max(np.sqrt(sol['u']**2 + sol['v']**2 + sol['w']**2))
        print 'z0 = %g, eps = %g, diff = %e' %(z0,eps,diff)

