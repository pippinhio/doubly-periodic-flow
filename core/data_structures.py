#!/usr/bin/env python

from __future__ import division

import numpy as np

def make_forces_struct(X0_star,F,par):
    # Two dictionaries with forces are computed.
    # forces_dom: 
    #   Contains only forces (and possibly image elements) that 
    #   correspond to locations in the main domain.
    # forces_cop:
    #   Contains periodic copies of forces (and possibly image elements). 
    #   These copies are needed for the loacal piece in the Ewald splitting
    #   and for the reference solution, which adds up Stokeslets in free
    #   space.
    nforces = len(F)
    forces_dom = {'nforces':nforces,'X0_star':X0_star,'F':F}

    images = par['images']
    if images:
        X0 = np.zeros(len(X0_star),dtype=[('x',float),('y',float),('z',float)])
        X0['x'] =  X0_star['x']
        X0['y'] =  X0_star['y']
        X0['z'] = -X0_star['z']
        Q = np.zeros(len(F),dtype=[('q1',float),('q2',float),('q3',float)])
        Q['q1'] = -F['f1']
        Q['q2'] = -F['f2']
        Q['q3'] =  F['f3']
        h  = X0_star['z']
        forces_dom.update({'X0':X0,'F':F,'Q':Q,'h':h})

    method = par['method']
    if method == 'FFT':
        forces = {'domain':forces_dom}  
    elif method in {'Real','Ewald'}:
        forces_cop = make_periodic_copies(forces_dom,par)   
        forces = {'domain':forces_dom,'copies':forces_cop}
    
    return forces


    
def make_periodic_copies(forces_dom,par):
    
    L_x = par['box']['L_x']
    L_y = par['box']['L_y']
    
    method = par['method']
    if method == 'Real':
        ncopies_Rx = par['reg']['ncopies_R']
        ncopies_Ry = par['reg']['ncopies_R']
    elif method == 'Ewald':
        r_cutoff = par['reg']['r_cutoff']   
        ncopies_Rx = int(np.ceil(r_cutoff/L_x))
        ncopies_Ry = int(np.ceil(r_cutoff/L_y))

    ncopies_x = 2*ncopies_Rx + 1
    ncopies_y = 2*ncopies_Ry + 1
    ncopies = ncopies_x * ncopies_y
    
    X0_star = forces_dom['X0_star']
    X0_star_cop = np.tile(X0_star,ncopies)     
    nforces = forces_dom['nforces']
    for nx in range(ncopies_x):
        for ny in range(ncopies_y):
             X0_star_cop[nforces*(nx*ncopies_y+ny):nforces*(nx*ncopies_y+ny+1)]['x'] += (nx - ncopies_Rx)*L_x
             X0_star_cop[nforces*(nx*ncopies_y+ny):nforces*(nx*ncopies_y+ny+1)]['y'] += (ny - ncopies_Ry)*L_y
    
    F = forces_dom['F']
    F_cop = np.tile(F,ncopies)

    # Delete fake copies that are too far away from the main domain
    if method == 'Ewald':
        x_a = par['box']['x_a']
        x_b = par['box']['x_b']
        y_a = par['box']['y_a']
        y_b = par['box']['y_b']
        x0_star_cop = X0_star_cop['x']
        y0_star_cop = X0_star_cop['y']
        idx = (  (x0_star_cop >= x_a-r_cutoff) 
               & (x0_star_cop <= x_b+r_cutoff)  
               & (y0_star_cop >= y_a-r_cutoff) 
               & (y0_star_cop <= y_b+r_cutoff)
              )
        X0_star_cop = X0_star_cop[idx]
        F_cop = F_cop[idx]

    forces_cop = {'nforces':len(X0_star_cop),'X0_star':X0_star_cop,'F':F_cop}
    
    images = par['images']
    if images:
        X0_cop = np.zeros(len(X0_star_cop),dtype=[('x',float),('y',float),('z',float)])
        X0_cop['x'] =  X0_star_cop['x']
        X0_cop['y'] =  X0_star_cop['y']
        X0_cop['z'] = -X0_star_cop['z']
        Q_cop = np.zeros(len(F_cop),dtype=[('q1',float),('q2',float),('q3',float)])
        Q_cop['q1'] = -F_cop['f1']
        Q_cop['q2'] = -F_cop['f2']
        Q_cop['q3'] =  F_cop['f3']
        h_cop  = X0_star_cop['z']
        forces_cop.update({'X0':X0_cop,'F':F_cop,'Q':Q_cop,'h':h_cop})
        
    return forces_cop

