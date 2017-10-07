#!/usr/bin/env python

from __future__ import division
import warnings

import numpy as np
from numpy import floor
from numpy.fft import ifft2
from multiprocessing import Pool

from velocity_fft import velocity_fft
from velocity_real import velocity_real, compute_plane


def velocity_from_forces(forces,par,n):
    grid = par['grid']
    xx = grid['x']
    yy = grid['y']
    zz = grid['z']
    kk = grid['k']
    mm = grid['m']
    N_x = grid['N_x']
    N_y = grid['N_y']
    N_z = grid['N_z']
    shape = grid['shape']
    
    if n > N_z:
        warnings.warn('Requested %d cores but there were only %d layers in z'
                      % (n,N_z))
        n = N_z
    
    # Certain debugging features only work when the code is 
    # exceuted in serial (e.g. code.interact(local=locals()).
    # It is therefore crucial that when only one process is 
    # requested, the code is run in serial and NOT as a 
    # parallel code with only 1 process.  
    if n == 1:
        return velocity_from_forces_serial([forces,par])
    else:
        u = np.zeros(shape)
        v = np.zeros(shape)
        w = np.zeros(shape)
        
        # assign processes to z-layers
        idx = np.sort(np.array(range(N_z)) % n)
        L = []
        
        for i in range(n):
            # create grid for i-th process
            xx_i = xx[:,:,idx==i]
            yy_i = yy[:,:,idx==i]
            zz_i = zz[:,:,idx==i]
            kk_i = kk[:,:,idx==i]
            mm_i = mm[:,:,idx==i]
            
            N_z_i = list(idx).count(i)

            grid_i = grid.copy()
            grid_i.update({'x':xx_i,'y':yy_i,'z':zz_i,'k':kk_i,'m':mm_i,
                'N_z':N_z_i,'shape':(N_x,N_y,N_z_i)})

            par_i = par.copy()
            par_i.update({'grid':grid_i})
            
            # The i-th element of list L is a list that contains all 
            # varibles handed to the i-th process 
            L.append([forces,par_i])
        
        # create processes
        pool = Pool(processes=n)
        res = pool.map(velocity_from_forces_serial,L)
        pool.close()
        pool.join()
        
        # organize results from all processes into one velocity array
        for i in range(n):
            u[:,:,idx==i] = res[i]['u']
            v[:,:,idx==i] = res[i]['v']
            w[:,:,idx==i] = res[i]['w']

        return {'u':u,'v':v,'w':w}



def velocity_from_forces_serial(L):
    forces = L[0]
    par = L[1]
    method = par['method']
    
    if method == 'Real':                 
        par.update({'splitting':False})
        (u,v,w) = compute_velocity('Real',forces['copies'],par)
        
        if not par['images']:
            # The correcting plane is only implemented for the 
            # Stokeslet. In particular, it is not implemented
            # for the method of images.
            pass
#            (plane_u,plane_v,plane_w) = compute_plane(forces['domain'],par)
#            u += plane_u
#            v += plane_v
#            w += plane_w
             
    elif method == 'FFT':
        par.update({'splitting':False})
        (u,v,w) = compute_velocity('FFT',forces['domain'],par)
        
    elif method == 'Ewald':
        par.update({'splitting':True})
        (u_erf ,v_erf ,w_erf ) = compute_velocity('FFT' ,forces['domain'],par)
        (u_erfc,v_erfc,w_erfc) = compute_velocity('Real',forces['copies'],par)
        u = u_erf + u_erfc
        v = v_erf + v_erfc
        w = w_erf + w_erfc
                 
    sol = {'u':u,'v':v,'w':w}
    return sol



def compute_velocity(method,forces,par):
    shape = par['grid']['shape']
    if method == 'Real':
        u = np.zeros(shape) 
        v = np.zeros(shape) 
        w = np.zeros(shape) 
    elif method == 'FFT':
        u_hat = np.zeros(shape,dtype=complex) 
        v_hat = np.zeros(shape,dtype=complex) 
        w_hat = np.zeros(shape,dtype=complex) 

    X0_star = forces['X0_star']
    F = forces['F'] 
    images = par['images']
    if images: 
        X0 = forces['X0']
        Q = forces['Q'] 
        h = forces['h']
     
    # compute velocity field for every single force
    nforces = forces['nforces']
    for i in range(nforces):    
        force = {'X0_star':X0_star[i],'F':F[i]}         
        if images:
            force.update({'X0':X0[i],'Q':Q[i],'h':h[i]})
        
        if method == 'Real':
            (ui,vi,wi) = velocity_real(force,par)           
            u += ui
            v += vi
            w += wi
        elif method == 'FFT':
            (ui_hat,vi_hat,wi_hat) = velocity_fft(force,par)
            u_hat += ui_hat
            v_hat += vi_hat
            w_hat += wi_hat
         
    if method == 'FFT':
        # scale velocities
        L_x = par['box']['L_x'] 
        L_y = par['box']['L_y']
        N_x = par['grid']['N_x'] 
        N_y = par['grid']['N_y']
        u_hat = u_hat*(N_x*N_y)/(L_x*L_y)
        v_hat = v_hat*(N_x*N_y)/(L_x*L_y)
        w_hat = w_hat*(N_x*N_y)/(L_x*L_y)
         
        # After taking the inverse Fourier Transform
        # the imaginary part of the solutions should 
        # be at the order of machine precision. 
        u = np.real(ifft2(u_hat,axes=(0,1)))
        v = np.real(ifft2(v_hat,axes=(0,1)))
        w = np.real(ifft2(w_hat,axes=(0,1)))
    
    return (u,v,w)
    
