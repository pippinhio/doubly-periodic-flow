#!/usr/bin/env python

from __future__ import division
import warnings

import numpy as np
from numpy import pi, sqrt, exp
from scipy.special import erf, erfc


def velocity_real(force,par): 
    # For Ewald splitting:
    # The grid variables xx_full, yy_full, zz_full contain the whole 
    # grid.  The velocity variables u_full, v_full, w_full refer to 
    # the whole grid.  The usual grid variables xx, yy, zz contain 
    # only those points of the full grid that are within r_cutoff of
    # the force location at (x0,y0,z0).  The velocity variables 
    # u, v, w are defined on this smaller grid.
    #
    # When used for the reference solution:
    # In this case, there is no distiction between the variables 
    # grid xx_full, yy_full, zz_full and xx, yy, zz. The velocity 
    # variables u_full, v_full, w_full are not needed since 
    # u, v, w contain the velocity on the whole grid.   
    grid = par['grid']
    xx_full = grid['x']
    yy_full = grid['y'] 
    zz_full = grid['z']
    
    X0_star = force['X0_star']
    x0_star = X0_star['x'] 
    y0_star = X0_star['y']
    z0_star = X0_star['z']
    
    r_star_full = sqrt((xx_full - x0_star)**2 + (yy_full - y0_star)**2 
                       + (zz_full - z0_star)**2)
    
    method = par['method']
    if method == 'Real':
        r_star = r_star_full
        xx = xx_full
        yy = yy_full
        zz = zz_full
    elif method == 'Ewald':
        # find neighbors
        r_cutoff = par['reg']['r_cutoff']
        neighbors_idx = (r_star_full<=r_cutoff)

        r_star = r_star_full[neighbors_idx]
        xx = xx_full[neighbors_idx]
        yy = yy_full[neighbors_idx]
        zz = zz_full[neighbors_idx]
     
    xx_tilde_star = xx - x0_star
    yy_tilde_star = yy - y0_star
    zz_tilde_star = zz - z0_star 
    F = force['F']
    f1 = F['f1'] 
    f2 = F['f2'] 
    f3 = F['f3']
    (H1_star,H2_star) = regularized_elements(r_star,par,'H')

    # Stokeslet_star
    u = f1*H1_star
    v = f2*H1_star
    w = f3*H1_star

    u += (f1*xx_tilde_star + f2*yy_tilde_star + f3*zz_tilde_star)*H2_star*xx_tilde_star
    v += (f1*xx_tilde_star + f2*yy_tilde_star + f3*zz_tilde_star)*H2_star*yy_tilde_star
    w += (f1*xx_tilde_star + f2*yy_tilde_star + f3*zz_tilde_star)*H2_star*zz_tilde_star
    
    images = par['images']
    if images:
        X0 = force['X0']
        x0 = X0['x'] 
        y0 = X0['y'] 
        z0 = X0['z']
        Q = force['Q']
        q1 = Q['q1']
        q2 = Q['q2'] 
        q3 = Q['q3']
        h = force['h']
        
        r = sqrt( (xx - x0 )**2 + (yy - y0 )**2 + (zz - z0 )**2)
        xx_tilde = xx - x0
        yy_tilde = yy - y0
        zz_tilde = zz - z0
        qmxm = q1*xx_tilde + q2*yy_tilde + q3*zz_tilde;

        (H1,H2,H1_prime,H2_prime,Gs_prime_over_r,Gd_prime_over_r,D1,D2) = \
            regularized_elements(r,par,'all')

        # Stokeslet
        u -= f1*H1
        v -= f2*H1
        w -= f3*H1

        u -= (f1*xx_tilde   + f2*yy_tilde   + f3*zz_tilde  )*H2*xx_tilde
        v -= (f1*xx_tilde   + f2*yy_tilde   + f3*zz_tilde  )*H2*yy_tilde
        w -= (f1*xx_tilde   + f2*yy_tilde   + f3*zz_tilde  )*H2*zz_tilde
        
        # Doublet
        u += 2*h*(H2*(zz_tilde*q1 + xx_tilde*q3) + H2_prime/r*qmxm*xx_tilde*zz_tilde                  )
        v += 2*h*(H2*(zz_tilde*q2 + yy_tilde*q3) + H2_prime/r*qmxm*yy_tilde*zz_tilde                  )
        w += 2*h*(H2*(zz_tilde*q3 + zz_tilde*q3) + H2_prime/r*qmxm*zz_tilde*zz_tilde + H1_prime/r*qmxm)

        # Dipole
        u += h**2*(D1*q1 + D2*qmxm*xx_tilde)
        v += h**2*(D1*q2 + D2*qmxm*yy_tilde)
        w += h**2*(D1*q3 + D2*qmxm*zz_tilde)

        # Rotlets
        u += 2*h*(Gd_prime_over_r - Gs_prime_over_r)*zz_tilde*q1               
        v += 2*h*(Gd_prime_over_r - Gs_prime_over_r)*zz_tilde*q2               
        w += 2*h*(Gd_prime_over_r - Gs_prime_over_r)*(-xx_tilde*q1 - yy_tilde*q2)
     
    if method == 'Real':
        return (u,v,w)
    if method == 'Ewald':
        # The solution was only computed inside a ball of radius
        # r_cutoff around the force location.  The solution is
        # now mapped back onto the original grid.
        u_full = np.zeros(grid['shape'])
        u_full[neighbors_idx] = u
        v_full = np.zeros(grid['shape'])
        v_full[neighbors_idx] = v
        w_full = np.zeros(grid['shape'])
        w_full[neighbors_idx] = w        
        return (u_full,v_full,w_full)


    
def regularized_elements(r,par,elements):
    # At first only elements H1 and H2 are computed. All other 
    # elements are computed subsequently only if the input 
    # variable elements is set to 'all'.  
    ep1 = par['reg']['epsilon']
    ep2 = ep1**2
    ep3 = ep1**3
    ep4 = ep1**4
    ep5 = ep1**5

    r1 = r # Note: this creates r1 only by reference
    r2 = r1**2
    r4 = r1**4
    roep1 = r1/ep1
    roep2 = r2/ep2

    # When r is small, we replace all expressions with their 
    # corresponding Taylor series approximation.  The Taylor 
    # series is implemented up to order 6.  This  means that, 
    # for instance, requiring r2/ep2<10^-5 gives an approximation 
    # exact up to 15 digits.  Note that the error in the 
    # Taylor series for xi is always smaller then the error 
    # for the series in epsilons since we require ep<xi.
    r_tiny_idx = (roep2 < 1e-05) 
    r_tiny = r[r_tiny_idx]
    r[r_tiny_idx] = np.nan # This avoids division by numerical zero.
    
    rm1 = 1/r
    rm2 = rm1**2
    rm3 = rm2*rm1

    coeff_exp_ep = 0.04489678053129164/ep1 # = 1/(4*pi^(3/2))*1/ep1
    ExpEp = exp(-roep2)
    coeff_erf    = 0.03978873577297383 # = 1/(8*pi)
    ErfEp = erf( roep1)

    coeff_taylor = 0.17958712212516656 # = 1/sqrt(pi^3)
    roep1_tiny = r_tiny/ep1
    roep2_tiny = roep1_tiny**2
    roep4_tiny = roep2_tiny**2

    # Compute elements.
    H1 =  coeff_exp_ep    *ExpEp + coeff_erf*rm1*ErfEp
    H2 = -coeff_exp_ep*rm2*ExpEp + coeff_erf*rm3*ErfEp

    # Compute Taylor series approximation for small r.
    H1_taylor = coeff_taylor/ep1*( 1/2 -  1/3*roep2_tiny + 3/20*roep4_tiny)
    H2_taylor = coeff_taylor/ep3*( 1/6 - 1/10*roep2_tiny + 1/28*roep4_tiny)

    splitting = par['splitting']
    if splitting:
        xi1  = par['reg']['xi']
        xi2 = xi1**2
        xi3 = xi1**3

        roxi1 = r1/xi1
        roxi2 = r2/xi2

        ExpXi = exp(-roxi2)
        coeff_exp_xi = 0.04489678053129164/xi1 # = 1/(4*pi^(3/2))*1/xi1
        ErfXi = erf( roxi1)

        # Compute elements
        H1 += coeff_exp_xi    *(-3 +  2*roxi2 )*ExpXi - coeff_erf*rm1*ErfXi
        H2 += coeff_exp_xi*rm2*( 1 -  2*roxi2 )*ExpXi - coeff_erf*rm3*ErfXi
        
        # Compute Taylor series approximation for small r
        roxi1_tiny = r_tiny/xi1
        roxi2_tiny = roxi1_tiny**2
        roxi4_tiny = roxi2_tiny**2
    
        H1_taylor += coeff_taylor/xi1*(   -1 +  4/3*roxi2_tiny - 9/10*roxi4_tiny)
        H2_taylor += coeff_taylor/xi3*( -2/3 +  3/5*roxi2_tiny -  2/7*roxi4_tiny)
    
    # Replace solution near the numerical singularity
    # with its Taylor series approximation. 
    H1[r_tiny_idx] = H1_taylor
    H2[r_tiny_idx] = H2_taylor

    if elements == 'all':
        rm4 = rm2**2
        rm5 = rm3*rm2
        roep3_tiny = roep2_tiny*roep1_tiny
        roep5_tiny = roep3_tiny*roep2_tiny

        # Compute elements
        H1_prime = coeff_exp_ep*rm1*( 1 - 2*roep2 )*ExpEp -   coeff_erf*rm2*ErfEp
        H2_prime = coeff_exp_ep*rm3*( 3 + 2*roep2 )*ExpEp - 3*coeff_erf*rm4*ErfEp
        Gs_prime = coeff_exp_ep*rm1*(-2 + 2*roep2 )*ExpEp + 2*coeff_erf*rm2*ErfEp
        Gd_prime = coeff_exp_ep*rm1*(-2           )*ExpEp + 2*coeff_erf*rm2*ErfEp
        D1       = coeff_exp_ep*rm2*( 2 + 4*roep2 )*ExpEp - 2*coeff_erf*rm3*ErfEp
        D2       = coeff_exp_ep*rm4*(-6 - 4*roep2 )*ExpEp + 6*coeff_erf*rm5*ErfEp

        # Compute Taylor series approximation for small r
        H1_prime_taylor        = coeff_taylor/ep2*(-2/3*roep1_tiny +  3/5*roep3_tiny -  2/7*roep5_tiny)
        H2_prime_taylor        = coeff_taylor/ep4*(-1/5*roep1_tiny +  1/7*roep3_tiny - 1/18*roep5_tiny)
        Gs_prime_over_r_taylor = coeff_taylor/ep2*( 5/6            - 7/10*roep2_tiny + 9/28*roep4_tiny)
        Gd_prime_over_r_taylor = coeff_taylor/ep2*( 1/3            -  1/5*roep2_tiny + 1/14*roep4_tiny)
        D1_taylor              = coeff_taylor/ep3*( 2/3            -  4/5*roep2_tiny +  3/7*roep4_tiny)
        D2_taylor              = coeff_taylor/ep5*( 2/5            -  2/7*roep2_tiny +  1/9*roep4_tiny)

        if splitting:
            xi4 = xi1**4
            xi5 = xi1**5
            roxi4 = r4/xi4
            roxi3_tiny = roxi2_tiny*roxi1_tiny
            roxi5_tiny = roxi3_tiny*roxi1_tiny
    
            # Compute elements
            H1_prime += coeff_exp_xi*rm1*(-1 + 10*roxi2 - 4*roxi4)*ExpXi +   coeff_erf*rm2*ErfXi
            H2_prime += coeff_exp_xi*rm3*(-3 -  2*roxi2 + 4*roxi4)*ExpXi + 3*coeff_erf*rm4*ErfXi
            Gs_prime += coeff_exp_xi*rm1*( 2 - 12*roxi2 + 4*roxi4)*ExpXi - 2*coeff_erf*rm2*ErfXi
            Gd_prime += coeff_exp_xi*rm1*( 2 -  4*roxi2          )*ExpXi - 2*coeff_erf*rm2*ErfXi
            D1       += coeff_exp_xi*rm2*(-2 - 12*roxi2 + 8*roxi4)*ExpXi + 2*coeff_erf*rm3*ErfXi
            D2       += coeff_exp_xi*rm4*( 6 +  4*roxi2 - 8*roxi4)*ExpXi - 6*coeff_erf*rm5*ErfXi
        
            # Compute Taylor series approximation for small r
            roxi1_tiny = r_tiny/xi1
            roxi2_tiny = roxi1_tiny**2
            roxi3_tiny = roxi2_tiny*roxi1_tiny
            roxi4_tiny = roxi2_tiny**2
            roxi5_tiny = roxi3_tiny*roxi1_tiny

            H1_prime_taylor        += coeff_taylor/xi2*(  8/3*roxi1_tiny - 18/5*roxi3_tiny + 16/7*roxi5_tiny)
            H2_prime_taylor        += coeff_taylor/xi4*(  6/5*roxi1_tiny -  8/7*roxi3_tiny +  5/9*roxi5_tiny)
            Gs_prime_over_r_taylor += coeff_taylor/xi2*(-10/3            + 21/5*roxi2_tiny - 18/7*roxi4_tiny)
            Gd_prime_over_r_taylor += coeff_taylor/xi2*( -4/3            +  6/5*roxi2_tiny -  4/7*roxi4_tiny)
            D1_taylor              += coeff_taylor/xi3*( -8/3            + 24/5*roxi2_tiny - 24/7*roxi4_tiny)
            D2_taylor              += coeff_taylor/xi5*(-12/5            + 16/7*roxi2_tiny - 10/9*roxi4_tiny)

        # This function returns the quantity G'/r, and not G' 
        # itself, since dividing by r later could mean dividing
        # by numerical zero.
        Gs_prime_over_r = Gs_prime*rm1
        Gd_prime_over_r = Gd_prime*rm1

        # Replace solution near the numerical singularity
        # with its Taylor series approximation. 
        H1_prime       [r_tiny_idx] = H1_prime_taylor
        H2_prime       [r_tiny_idx] = H2_prime_taylor
        Gs_prime_over_r[r_tiny_idx] = Gs_prime_over_r_taylor
        Gd_prime_over_r[r_tiny_idx] = Gd_prime_over_r_taylor
        D1             [r_tiny_idx] = D1_taylor
        D2             [r_tiny_idx] = D2_taylor

    if elements == 'H':
        return (H1,H2)
    elif elements == 'all':
        return (H1,H2,H1_prime,H2_prime,Gs_prime_over_r,Gd_prime_over_r,D1,D2) 
        

        
def compute_plane(forces,par):
    # The plane can be computed in order to improve the reference
    # solution only when there are no images.
    L_x = par['box']['L_x']
    L_y = par['box']['L_x']
    if L_x == L_y:
        L = L_x
    else:
        warnings.warn('Plane is only defined for quadratic domain '
                      + '(i.e. L_x = L_y')

    partial_sum_S1 = 0 
    partial_sum_S2 = 0 
    partial_sum_S3 = 0 
    partial_sum_S4 = 0 

    # Note: The plane was derived for Mx = My only.
    Mx = par['reg']['ncopies_R'] 
    My = par['reg']['ncopies_R'] 
    for m in range(-Mx,Mx+1):
        for n in range(-My,My+1):
            if not (m == 0 and n == 0):
                n2 = n**2
                n4 = n**4
                m2 = m**2
                partial_sum_S1 +=     1 /(m2+n2)**(3/2)
                partial_sum_S2 +=    n2 /(m2+n2)**(5/2)
                partial_sum_S3 +=    n4 /(m2+n2)**(7/2)
                partial_sum_S4 += n2*m2 /(m2+n2)**(7/2)
                             
    S1 = 1/(8*pi*L**3)*( 9.033621683100950 - partial_sum_S1 )
    S2 = 1/(8*pi*L**3)*( 4.516810841550475 - partial_sum_S2 )
    S3 = 1/(8*pi*L**3)*( 3.745708094289508 - partial_sum_S3 )
    S4 = 1/(8*pi*L**3)*( 0.771102747260967 - partial_sum_S4 )

    X0_star = forces['X0_star'] 
    F = forces['F']
    nforces = forces['nforces']
    
    c1 = 0
    cx = 0 
    cy = 0
    cz = 0

    for i in range(nforces):  
        f1 = F[i]['f1']
        f2 = F[i]['f2']
        f3 = F[i]['f3']

        xk = X0_star[i]['x']
        yk = X0_star[i]['y']
        zk = X0_star[i]['z']
        xk2 = xk**2
        yk2 = yk**2
        zk2 = zk**2

        var1 = np.array([xk, yk, zk])
        var2 = np.array([xk2, xk*yk, xk*zk, yk*yk, yk*zk, zk2])
        
        A11 = np.array([
            [ 0.5*f1,  1.0*f2,  1.0*f3, -0.5*f1,     0.0, -0.5*f1],
            [-0.5*f2,  1.0*f1,     0.0,  0.5*f2,  1.0*f3, -0.5*f2],
            [-0.5*f3,     0.0,  1.0*f1, -0.5*f3,  1.0*f2,  0.5*f3]
        ])
        A12 = np.array([
            [-6.0*f1, -6.0*f2, -3.0*f3,     0.0,     0.0, -1.5*f1],
            [    0.0, -6.0*f1,     0.0, -6.0*f2, -3.0*f3, -1.5*f2],
            [ 1.5*f3,     0.0, -3.0*f1,  1.5*f3, -3.0*f2,     0.0]
        ])
        A13 = np.array([
            [ 7.5*f1,     0.0,     0.0,     0.0,     0.0,     0.0],
            [    0.0,     0.0,     0.0,  7.5*f2,     0.0,     0.0],
            [    0.0,     0.0,     0.0,     0.0,     0.0,     0.0]
        ])
        A14 = np.array([
            [    0.0, 15.0*f2,     0.0,  7.5*f1,     0.0,     0.0],
            [ 7.5*f2, 15.0*f1,     0.0,     0.0,     0.0,     0.0],
            [    0.0,     0.0,     0.0,     0.0,     0.0,     0.0]
        ])
                                                                                    
        Ax1 = np.array([
            [ -1.0*f1,  -1.0*f2,  -1.0*f3],
            [  1.0*f2,  -1.0*f1,      0.0],
            [  1.0*f3,      0.0,  -1.0*f1]
        ])
        Ax2 = np.array([ 
            [ 12.0*f1,   6.0*f2,   3.0*f3],
            [     0.0,   6.0*f1,      0.0],
            [ -3.0*f3,      0.0,   3.0*f1]
        ])           
        Ax3 = np.array([
            [-15.0*f1,      0.0,      0.0],
            [     0.0,      0.0,      0.0],
            [     0.0,      0.0,      0.0]
        ])           
        Ax4 = np.array([
            [     0.0, -15.0*f2,      0.0],
            [-15.0*f2, -15.0*f1,      0.0],
            [     0.0,      0.0,      0.0]
        ])

        Ay1 = np.array([
            [ -1.0*f2,   1.0*f1,      0.0],
            [ -1.0*f1,  -1.0*f2,  -1.0*f3],
            [     0.0,   1.0*f3,  -1.0*f2]
        ])
        Ay2 = np.array([                
            [  6.0*f2,      0.0,      0.0],
            [  6.0*f1,  12.0*f2,   3.0*f3],
            [     0.0,  -3.0*f3,   3.0*f2]
        ])           
        Ay3 = np.array([
            [     0.0,      0.0,      0.0],
            [     0.0, -15.0*f2,      0.0],
            [     0.0,      0.0,      0.0]
        ])           
        Ay4 = np.array([
            [-15.0*f2, -15.0*f1,      0.0],
            [-15.0*f1,      0.0,      0.0],
            [     0.0,      0.0,      0.0]
        ])

        Az1 = np.array([
            [ -1.0*f3,      0.0,   1.0*f1],
            [     0.0,  -1.0*f3,   1.0*f2],
            [ -1.0*f1,  -1.0*f2,  -1.0*f3]
        ])
        Az2 = np.array([                
            [  3.0*f3,      0.0,   3.0*f1],
            [     0.0,   3.0*f3,   3.0*f2],
            [  3.0*f1,   3.0*f2,      0.0]
        ])           
        Az3 = np.array([
            [     0.0,      0.0,      0.0],
            [     0.0,      0.0,      0.0],
            [     0.0,      0.0,      0.0]
        ])           
        Az4 = np.array([
            [     0.0,      0.0,      0.0],
            [     0.0,      0.0,      0.0],
            [     0.0,      0.0,      0.0]
        ])

        c1 += np.dot(S1*A11 + S2*A12 + S3*A13 + S4*A14, var2)
        cx += np.dot(S1*Ax1 + S2*Ax2 + S3*Ax3 + S4*Ax4, var1)
        cy += np.dot(S1*Ay1 + S2*Ay2 + S3*Ay3 + S4*Ay4, var1)
        cz += np.dot(S1*Az1 + S2*Az2 + S3*Az3 + S4*Az4, var1)

    xx = par['grid']['x']
    yy = par['grid']['y']
    zz = par['grid']['z']
    plane_u = c1[0] + cx[0]*xx + cy[0]*yy + cz[0]*zz
    plane_v = c1[1] + cx[1]*xx + cy[1]*yy + cz[1]*zz
    plane_w = c1[2] + cx[2]*xx + cy[2]*yy + cz[2]*zz
    
    return (plane_u,plane_v,plane_w)
    
