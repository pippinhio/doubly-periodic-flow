#!/usr/bin/env python
from __future__ import division

import numpy as np
from numpy import sqrt
from numpy import random as rd
import time

from src.data_structures import make_forces_struct
from src.velocity.velocity_from_forces import velocity_from_forces


def run(X0,F,par,cores):
    forces = make_forces_struct(X0,F,par)
    start = time.time()
    sol = velocity_from_forces(forces,par,cores)
    elapsed = time.time()-start
    print('%s ran for %f seconds' % (par['method'],elapsed))
    return sol


def max_diff(sol1,sol2):
    u1 = sol1['u']
    v1 = sol1['v']
    w1 = sol1['w']
    u2 = sol2['u']
    v2 = sol2['v']
    w2 = sol2['w']

    diff = sqrt( (u1-u2)**2 + (v1-v2)**2 + (w1-w2)**2 )
    return np.amax(diff)


def max_velo_at_wall(sol):
    u = sol['u']
    v = sol['v']
    w = sol['w']
    r = sqrt( u**2 + v**2 + w**2 )
    res = np.amax(r[:,:,0])     
    return res


def velo_at_wall(sol,par):
    u0 = sol['u'][:,:,0]
    v0 = sol['v'][:,:,0]
    w0 = sol['w'][:,:,0]
    
    r = sqrt( u0**2 + v0**2 + w0**2)
    print('Maximum velocity at wall: %e' % np.amax(r))
