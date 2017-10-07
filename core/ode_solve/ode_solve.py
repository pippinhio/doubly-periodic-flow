#!/usr/bin/env python

from __future__ import division

import numpy as np

def adams_bashforth(Var,F,dt,order):
    # Integrates the ODE Var'=F(Var) for one time step of size dt using 
    # Adams-Bashforth method.  Choices for order are order=1,2,3,4.
    
    (var_x,f_u) = adams_bashforth_1D(Var['x'],F['u'],dt,order)
    (var_y,f_v) = adams_bashforth_1D(Var['y'],F['v'],dt,order)
    (var_z,f_w) = adams_bashforth_1D(Var['z'],F['w'],dt,order)
#    Var.update({'x':var_x,'y':var_y,'z':var_z})
    Var['x'] = var_x
    Var['y'] = var_y
    Var['z'] = var_z
        
    F.update({'u':f_u,'v':f_v,'w':f_w})
    
    return (Var,F)



def adams_bashforth_1D(y,f,dt,order):

    if order == 1:    
        # Euler's method
        y = y + dt*f['n']
#        y += dt*f['n']
        f.update({'nm1':f['n']})

    elif order == 2:
        y = y + dt*(3/2*f['n'] - 1/2*f['nm1'])
#        y += dt*(3/2*f['n'] - 1/2*f['nm1'])
        f.update({'nm2':f['nm1']})
        f.update({'nm1':f['n']})

    elif order == 3:
        y = y + dt*(23/12*f['n'] - 4/3*f['nm1'] + 5/12*f['nm2'])
#        y += dt*(23/12*f['n'] - 4/3*f['nm1'] + 5/12*f['nm2'])
        f.update({'nm3':f['nm2']})
        f.update({'nm2':f['nm1']})
        f.update({'nm1':f['n']})

    elif order == 4:
        y = y + dt*(55/24*f['n'] - 59/24*f['nm1'] + 37/24*f['nm2'] - 3/8*f['nm3'])
#        y += dt*(55/24*f['n'] - 59/24*f['nm1'] + 37/24*f['nm2'] - 3/8*f['nm3'])
        f.update({'nm3':f['nm2']})
        f.update({'nm2':f['nm1']})
        f.update({'nm1':f['n']})

    return (y,f)
