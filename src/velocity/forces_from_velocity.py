#!/usr/bin/python

from __future__ import division
import time
import copy

import numpy as np
from numpy import linalg as LA

from velocity_from_forces import velocity_from_forces
from src.evaluation import interp
from src.grid import make_grid
from src.data_structures import make_forces_struct

def forces_from_velocity(X0,U,par,cores):
    # In the z-direction, the given boundaries of the computational
    # domain are irrellevant and instead the grid needs to encompass 
    # (only) the given forces.  Also there are too options for the 
    # grid spacing in z.
    # Option 1 (few forces):
    # Grid contains exactly those z-layers with a force on them.
    # Option 2 (many forces):
    # Use a uniform grid in z and interpolate to the force location
    # in z.
    start = time.time()

    par_solve = copy.deepcopy(par)
    # Define the new domain.
    box = par_solve['box']
    z0 = np.unique(X0['z']) # unique sorts the array and keeps unique elements
    z_layers_a = z0[0]
    z_layers_b = z0[-1]
    z_layers_L = z_layers_b - z_layers_a
    box.update({'z_a':z_layers_a,'z_b':z_layers_b,'L_z':z_layers_L})
    
    # Option 1.
    uniform_in_z = par_solve['grid']['uniform_in_z']
    if uniform_in_z:
        dz = par_solve['grid']['dz']  
        Nz_uniform = np.ceil((z_layers_b - z_layers_a)/dz)
    else:
        # In this case a uniform grid can't be used since dz is unknown
        Nz_uniform = np.inf
        
    # Option 2.
    Nz_nonuniform = len(z0)
    
    # Find new grid.
    dx = par_solve['grid']['dx']
    dy = par_solve['grid']['dy']
    if (uniform_in_z) and (1 < Nz_uniform) and (Nz_uniform < Nz_nonuniform):
        spacing = {'dx':dx,'dy':dy,'dz':dz,'uniform_in_z':True}
    else:
        spacing = {'dx':dx,'dy':dy,'z_layers':z0,'uniform_in_z':False}
    grid = make_grid(box,spacing)
    par_solve.update({'box':box,'grid':grid})

    # The matrix S decribes the linear relationship between forces and 
    # velocities: u = S*f. The forces are found by solving the system.
    nforces = len(X0)
    if np.all(U['v'] == 0) and is_constant(X0['y']):
        # This is the case when the planar cilia motion from
        # 'Sleigh' or 'SandersonSeigh' is used.
        force_dim = 2
        force_components = ['f1','f3']
        S = np.zeros((2*nforces,2*nforces))
        U_vec = np.concatenate((U['u'],U['w']))
    else:
        force_dim = 3
        force_components = ['f1','f2','f3']
        S = np.zeros((3*nforces,3*nforces))        
        U_vec = np.concatenate((U['u'],U['v'],U['w']))
    par.update({'force_dim':force_dim})

    counter = 0
    # SET UP MATRIX OLD
    for j in force_components:
        for i in range(nforces):
            # set unit force
            ej = np.zeros(1,dtype=[('f1',float),('f2',float),('f3',float)])
            ej[j] = 1
            # Note: X0[i] gives only the row (*,*,*) of type 'numpy.void'
            # Instead, X0[i:i+1] gives the structured array with (*,*,*)
            # as the only element.
            fj = make_forces_struct(X0[i:i+1],ej,par_solve)
            sol = velocity_from_forces(fj,par_solve,cores)
 
            u_X0 = interp(sol['u'],X0,par_solve)
            w_X0 = interp(sol['w'],X0,par_solve)
            if force_dim == 2:
                S[:,counter] = np.concatenate((u_X0,w_X0))            
            elif force_dim == 3:
                v_X0 = interp(sol['v'],X0,par_solve)
                S[:,counter] = np.concatenate((u_X0,v_X0,w_X0))            
            counter += 1

    print('Condition number = %e' %LA.cond(S))
    F_vec = LA.solve(S,U_vec)
    print 'Residual error = %g' % np.amax(np.abs(np.dot(S,F_vec)-U_vec))
    elapsed = time.time()-start
    print('System solve took %f seconds' % elapsed)
    
    F = np.zeros(nforces,dtype=[('f1',float),('f2',float),('f3',float)])
    F['f1'] = F_vec[0:nforces]
    if force_dim == 2:
        F['f2'] = 0
        F['f3'] = F_vec[nforces:2*nforces]        
    elif force_dim == 3:
        F['f2'] = F_vec[nforces:2*nforces]
        F['f3'] = F_vec[2*nforces:3*nforces]    
    return F

def is_constant(vec):
    return np.all(vec[1:]==vec[-1:])
