#!/usr/bin/python

from __future__ import division
import time
import copy

import numpy as np
from numpy import linalg as LA
from multiprocessing import Pool

from velocity_from_forces import velocity_from_forces
from src.evaluation import map_into_main_domain
from src.grid import make_grid
from src.data_structures import make_forces_struct

def velocity_from_forces_exact(forces,X_eval,par,cores):
    #X_eval needs to be a vector. If X_eval is a grid, use velocity_from_forces
    # and, if necessary, interpolate.
    idx = np.sort(np.array(range(len(X_eval))) % cores)
    L = []
    for i in range(cores):
        L.append([forces, X_eval[idx==i], par.copy()])
    if cores == 1:
        res = evaluate_at_point(L[0])
    else:
        pool = Pool(processes=cores)
        res = pool.map(evaluate_at_point, L)
        pool.close()
        pool.join()
    return np.hstack(res)

def evaluate_at_point(L):
    forces = L[0]
    X_eval = L[1]
    par = L[2]

    U_eval = np.zeros(len(X_eval),dtype=[('u',float),('v',float),('w',float)])
    for j in range(len(X_eval)):
        # Find grid with one layer at zj.
        spacing = {
            'dx':par['grid']['dx'],
            'dy':par['grid']['dy'],
            'z_layers':X_eval[j]['z'],
            'uniform_in_z':False
            }
        grid = make_grid(par['box'],spacing)
        par_solve = copy.deepcopy(par)
        par_solve.update({'grid':grid})

        # Shift grid so that Xj = (xj,yj,zj) is at (0,0,zj)
        X0 = copy.deepcopy(forces['domain']['X0_star'])
        X0['x'] = map_into_main_domain(X0['x']-X_eval[j]['x'], par['box'], 'x')
        X0['y'] = map_into_main_domain(X0['y']-X_eval[j]['y'], par['box'], 'y')
#        X0['x'] = X0['x']-X_eval[j]['x']
#        X0['y'] = X0['y']-X_eval[j]['y']

        forces_shifted = make_forces_struct(X0,forces['domain']['F'],par_solve)
        sol = velocity_from_forces(forces_shifted,par_solve,1)
        U_eval[j]['u'] = sol['u'][np.all([par_solve['grid']['x']==0,par_solve['grid']['y']==0],axis=0)][0]
        U_eval[j]['v'] = sol['v'][np.all([par_solve['grid']['x']==0,par_solve['grid']['y']==0],axis=0)][0]
        U_eval[j]['w'] = sol['w'][np.all([par_solve['grid']['x']==0,par_solve['grid']['y']==0],axis=0)][0]
    return U_eval
