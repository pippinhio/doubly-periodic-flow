#!/usr/bin/env python

from __future__ import division

import numpy as np
from numpy import sqrt
from numpy import random as rd
import unittest

# This is needed to that packages form the parent directory can be
# loaded without using relative imports.
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from src.evaluation import interp
from src.set_parameters import set_parameters
from help_routines import run, max_velo_at_wall, max_diff
from src.velocity.forces_from_velocity import forces_from_velocity

__author__ = "Franz Hoffmann"
__copyright__ = "Copyright 2014, Franz Hoffmann"
__credits__ = ["Franz Hoffmann", "Ricardo Cortez"]
__license__ = "GPL"
__version__ = "1.1.0 - Thesis"
__maintainer__ = "Franz Hoffmann"
__email__ = "fhoffma@tulane.edu"
__status__ = "Prototype"


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.point_type = [('x',float),('y',float),('z',float)]
        self.force_type = [('f1',float),('f2',float),('f3',float)]
        self.velo_type = [('u',float),('v',float),('w',float)]

        # When debugging the code, cores should be set to 1 since,
        # for instance, code.interact() only works when the code
        # is executed in serial.  Also, there is a bug in python 2.7
        # where a running programm that uses multiple processes can
        # not be interrupted with Crtl+C.
        self.cores = 4
        self.tol = 1e-14

        rd.seed(1)
        self.nforces = 10
        # The forces are located at points in [0,1]x[0,1]x[eps,1].
        # We don't want points too close to the wall at z=0 where a
        # zero velocity is enforced.
        X0 = np.zeros(self.nforces,dtype=self.point_type)
        X0['x'] = rd.rand(self.nforces)
        X0['y'] = rd.rand(self.nforces)
        X0['z'] = 0.99*rd.rand(self.nforces) + 0.01
        self.X0 = X0
        # forces in [-1,1]^3
        F = np.zeros(self.nforces,dtype=self.force_type)
        F['f1'] = 2*rd.rand(self.nforces) - 1
        F['f2'] = 2*rd.rand(self.nforces) - 1
        F['f3'] = 2*rd.rand(self.nforces) - 1
        self.F  = F

    # ------------------------------------------------
    # TESTS
    # ------------------------------------------------

    def test1_splitting_no_images(self):
        print ' '
        print 'test_splitting_no_images'

        # FFT
        par_FFT = set_parameters(method='FFT',epsilon='4dx',images=False)
        sol_FFT = run(self.X0,self.F,par_FFT,self.cores)
        # Ewald
        par_Ewald = set_parameters(method='Ewald',epsilon='4dx',xi='6dx',
                        images=False)
        sol_Ewald = run(self.X0,self.F,par_Ewald,self.cores)

        res = max_diff(sol_FFT,sol_Ewald)
        print('maximum difference between methods: %e' % res)
        self.assertLess(res,self.tol)


    def test2_reference_solution_no_images(self):
        print ' '
        print 'test_reference_solution_no_images'

        # This time intensive test is run with fewer forces.
        nforces_test = int(np.ceil(self.nforces/2))
        self.X0 = self.X0[0:nforces_test]
        self.F  = self.F[0:nforces_test]

        # Set net force to zero.
        self.F['f1'][-1] = -sum(self.F['f1'][:-1])
        self.F['f2'][-1] = -sum(self.F['f2'][:-1])
        self.F['f3'][-1] = -sum(self.F['f3'][:-1])

        # Real
        par_Real = set_parameters(method='Real',dz=1/11,epsilon='2dx',
                                  ncopies_R=20,images=False)
        sol_Real = run(self.X0,self.F,par_Real,self.cores)

        # Ewald
        par_Ewald = set_parameters(method='Ewald',dz=1/11,epsilon='2dx',
                                   xi='4dx',images=False)
        sol_Ewald = run(self.X0,self.F,par_Ewald,self.cores)

        res = max_diff(sol_Real,sol_Ewald)
        print('maximum difference between methods: %e' % res)
        self.assertLess(res,1e-04)


    def test3_no_slip(self):
        print ' '
        print 'test_no_slip'

        # Real
        par_Real = set_parameters(method='Real',epsilon='1dx',ncopies_R=0,
                                  images=True)
        sol_Real = run(self.X0,self.F,par_Real,self.cores)
        res = max_velo_at_wall(sol_Real)
        print('maximum velocity at wall: %e' % res)
        self.assertLess(res,self.tol)

        # FFT
        par_FFT = set_parameters(method='FFT',epsilon='4dx',images=True)
        sol_FFT = run(self.X0,self.F,par_FFT,self.cores)
        res = max_velo_at_wall(sol_FFT)
        print('maximum velocity at wall: %e' % res)
        self.assertLess(res,self.tol)

        # Ewald
        par_Ewald = set_parameters(method='Ewald',epsilon='1dx',xi='4dx',
                                   images=True)
        sol_Ewald = run(self.X0,self.F,par_Ewald,self.cores)
        res = max_velo_at_wall(sol_Ewald)
        print('maximum velocity at wall: %e' % res)
        self.assertLess(res,self.tol)


    def test4_splitting_images(self):
        print ' '
        print 'test_splitting_images'

        # FFT
        par_FFT = set_parameters(method='FFT',epsilon='4dx',images=True)
        sol_FFT = run(self.X0,self.F,par_FFT,self.cores)

        # Ewald
        par_Ewald = set_parameters(method='Ewald',epsilon='4dx',xi='6dx',
                                   images=True)
        sol_Ewald = run(self.X0,self.F,par_Ewald,self.cores)

        res = max_diff(sol_FFT,sol_Ewald)
        print('maximum difference between methods: %e' % res)
        self.assertLess(res,self.tol)


    def test5_interpolation_onto_non_uniform_grid(self):
        print ' '
        print 'test_interpolation_onto_non_uniform_grid'

        # This time intensive test is run with fewer forces.
        nforces_test = int(np.ceil(self.nforces/2))
        self.X0 = self.X0[0:nforces_test]
        self.F  = self.F[0:nforces_test]

        # Coarser grid.
        par1 = set_parameters(dx=1/64,dy=1/64,
                              z_layers=self.X0['z'],
                              method='Ewald',epsilon=4/128,xi=8/128,
                              images=True)
        sol1 = run(self.X0,self.F,par1,self.cores)
        grid1 = par1['grid']
        u1 = sol1['u']
        v1 = sol1['v']
        w1 = sol1['w']

        # Finer grid.
        par2 = set_parameters(dx=1/128,dy=1/128,dz=1/256,
                              method='FFT',epsilon=4/128,
                              images=True)
        sol2 = run(self.X0,self.F,par2,self.cores)
        u2 = interp(sol2['u'],grid1,par2)
        v2 = interp(sol2['v'],grid1,par2)
        w2 = interp(sol2['w'],grid1,par2)

        diff = sqrt( (u1-u2)**2 + (v1-v2)**2 + (w1-w2)**2 )
        res = np.amax(diff)
        print('maximum difference on coarse grid: %e' % res)
        self.assertLess(res,1e-04)


    def test6_non_square_domain(self):
        print ' '
        print 'test_non_square_domain'

        x_a = 1.0
        x_b = 2.5
        y_a = 0.5
        y_b = 1.3
        z_a = 0.0
        z_b = 0.8
        domain = [x_a,x_b,y_a,y_b,z_a,z_b]

        rd.seed(2)
        X0 = np.zeros(self.nforces,dtype=self.point_type)
        X0['x'] = (x_b-x_a)*rd.rand(self.nforces) + x_a
        X0['y'] = (y_b-y_a)*rd.rand(self.nforces) + y_a
        X0['z'] = (z_b-z_a)*rd.rand(self.nforces) + z_a

        F = np.zeros(self.nforces,dtype=self.force_type)
        F['f1'] = 2*rd.rand(self.nforces) - 1
        F['f2'] = 2*rd.rand(self.nforces) - 1
        F['f3'] = 2*rd.rand(self.nforces) - 1

        # FFT
        par_FFT = set_parameters(domain=domain,
                                 method='FFT',epsilon='4dx',
                                 images=True)
        sol_FFT = run(X0,F,par_FFT,self.cores)

        # Ewald
        par_Ewald = set_parameters(domain=domain,
                                   method='Ewald',epsilon='4dx',xi='6dx',
                                   images=True)
        sol_Ewald = run(X0,F,par_Ewald,self.cores)

        res = max_diff(sol_FFT,sol_Ewald)
        print('maximum difference between methods: %e' % res)
        self.assertLess(res,self.tol)


    def test7_solve_for_forces(self):
        print ' '
        print 'solve_for_forces'

        rd.seed(3)

        U = np.zeros(self.nforces,dtype=self.velo_type)
        U['u'] = 2*rd.rand(self.nforces) - 1
        U['v'] = 2*rd.rand(self.nforces) - 1
        U['w'] = 2*rd.rand(self.nforces) - 1

        par = set_parameters(method='Ewald',epsilon='1dx',xi='4dx',images=True)
        F = forces_from_velocity(self.X0,U,par,self.cores)
        par = set_parameters(method='Ewald',epsilon='1dx',xi='4dx',images=True,
                             z_layers=self.X0['z'])
        sol = run(self.X0,F,par,self.cores)
        up = interp(sol['u'],self.X0,par)
        vp = interp(sol['v'],self.X0,par)
        wp = interp(sol['w'],self.X0,par)
        ue = U['u']
        ve = U['v']
        we = U['w']
        diff = sqrt( (ue-up)**2 + (ve-vp)**2 + (we-wp)**2 )
        res = np.amax(diff) # use real L2 difference
        print('maximum difference in computed velocity: %e' % res)
        self.assertLess(res,self.tol)


    def test8_solve_for_forces_planar(self):
        print ' '
        print 'solve_for_forces_planar'

        rd.seed(3)

        U = np.zeros(self.nforces,dtype=self.velo_type)
        U['u'] = 2*rd.rand(self.nforces) - 1
        U['v'] = 0
        U['w'] = 2*rd.rand(self.nforces) - 1

        self.X0['y'] = 0.5

        par = set_parameters(method='Ewald',epsilon='1dx',xi='4dx',images=True)
        F = forces_from_velocity(self.X0,U,par,self.cores)
        par = set_parameters(method='Ewald',epsilon='1dx',xi='4dx',images=True,
                             z_layers=self.X0['z'])
        sol = run(self.X0,F,par,self.cores)
        up = interp(sol['u'],self.X0,par)
        vp = interp(sol['v'],self.X0,par)
        wp = interp(sol['w'],self.X0,par)
        ue = U['u']
        ve = U['v']
        we = U['w']
        diff = sqrt( (ue-up)**2 + (ve-vp)**2 + (we-wp)**2 )
        res = np.amax(diff) # use real L2 difference
        print('maximum difference in computed velocity: %e' % res)
        self.assertLess(res,self.tol)


if __name__ == '__main__':
    unittest.main()
