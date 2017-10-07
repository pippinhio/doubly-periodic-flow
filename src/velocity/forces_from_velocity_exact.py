#!/usr/bin/python

from __future__ import division
import time

import numpy as np
from numpy import linalg as LA

from src.data_structures import make_forces_struct
from velocity_from_forces_exact import velocity_from_forces_exact

def forces_from_velocity_exact(X0,U,par,cores):
    # The matrix S decribes the linear relationship between forces and 
    # velocities: u = S*f. The forces are found by solving the system.
    nforces = len(X0)
    S = np.zeros((3*nforces,3*nforces))
    U_vec = np.concatenate((U['u'],U['v'],U['w']))

    # set unit force
    ek = np.zeros(3,dtype=[('f1',float),('f2',float),('f3',float)])
    ek['f1'] = np.array([1,0,0])
    ek['f2'] = np.array([0,1,0])
    ek['f3'] = np.array([0,0,1])

    start = time.time()
    for k in range(3):
        for i in range(nforces):
            # Note: X0[i] gives only the row (*,*,*) of type 'numpy.void'
            # Instead, X0[i:i+1] gives the structured array with (*,*,*)
            # as the only element.
            sol = velocity_from_forces_exact(make_forces_struct(X0[i:i+1],ek[k:k+1],par),X0,par,cores)     
            S[i+nforces*k,:] = np.hstack([sol['u'],sol['v'],sol['w']])
    print('Condition number = %e' %LA.cond(S))
    F_vec = LA.solve(S,U_vec)
    print 'Residual error = %g' % np.amax(np.abs(np.dot(S,F_vec)-U_vec))
    print('System solve took %f seconds' % (time.time()-start))
    
    F = np.zeros(nforces,dtype=[('f1',float),('f2',float),('f3',float)])
    F['f1'] = F_vec[0:nforces]
    F['f2'] = F_vec[nforces:2*nforces]
    F['f3'] = F_vec[2*nforces:3*nforces]        
    return F


#    import code
#    code.interact(local=locals())  

