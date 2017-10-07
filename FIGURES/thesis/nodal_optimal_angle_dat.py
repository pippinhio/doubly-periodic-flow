#!/usr/bin/env python

from __future__ import division
import time
import os
import pickle

import numpy as np
from numpy import pi as pi
from numpy import sqrt as sqrt

# This is needed to that packages form the parent directory can be
# loaded without using relative imports.
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from core.velocity.forces_from_velocity import forces_from_velocity
from core.velocity.forces_from_velocity_exact import forces_from_velocity_exact
from core.velocity.velocity_from_forces import velocity_from_forces
from core.set_parameters import set_parameters

beat_pattern = 'nodal'
if beat_pattern == 'nodal':
    from core.cilia.nodal import parametrize

''' All files are saved in the temporary directory
     results_tmp.  Files that one intends to keep
     permanently should be moved to another directory
     in order to avoid that they are overwritten.'''

cores = 4
par = set_parameters(domain=[-0.5,0.5,-0.5,0.5,0.0,3.0],dx=1/64,dy=1/64,dz=1/64,epsilon=0.05,images=True)

L_x = par['box']['L_x']
L_y = par['box']['L_y']

T = 1 # Total time for one full beat cycle.
nsteps = 64 # Number of steps for one full cycle.
dt = T/nsteps
npoints = 5 # Number of discretization points for each cilium.
L = 1.0
ncilia_x = 1
ncilia_y = 1

cilia_config = {
    'T':T,
    'npoints':npoints,
    'ncilia_x':ncilia_x,
    'ncilia_y':ncilia_y,    
    'nsteps':nsteps,
    'dt':dt,
    'beat_pattern':beat_pattern,
    'L':L,
#    'theta':theta*pi/180,
#    'psi':psi*pi/180
}

output = []

for psi in [60,55,45]:
    for theta in range(0,90-psi,5):
        cilia_config.update({'theta':theta*pi/180,'psi':psi*pi/180})        
        netflow = {'u':0,'v':0}                    
        t = 0
        for i in range(nsteps):
            print('step %d' % i)
            (X0,U) = parametrize(t,cilia_config,par)[0:2]
            F = forces_from_velocity_exact(X0,U,par,cores)
            z0 = X0['z']

            netflow['u'] += np.sum(1/(L_x*L_y)*z0*F['f1']*dt)
            netflow['v'] += np.sum(1/(L_x*L_y)*z0*F['f2']*dt)

            t += dt

        print 'psi = %g, theta = %g' % (psi,theta)
        print 'netflow in u: %e' % netflow['u']
        print 'netflow in v: %e' % netflow['v']
        output.append(['psi=%g,theta=%g,netfow=%e' % (psi,theta,netflow['u'])])
for element in output:
    print element
