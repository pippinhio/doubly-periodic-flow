#!/usr/bin/env python

from __future__ import division

import numpy as np
from numpy import sqrt, exp
from scipy.special import erf, erfc


def velocity_fft(force,par):

    images = par['images']
    if not images:
        green_s_star = greens_function('s',force['X0_star'],par)
        [u_hat_Sstar,v_hat_Sstar,w_hat_Sstar] = velocity_FFT_element('Stokeslet',force,green_s_star,par)

        u_hat = u_hat_Sstar
        v_hat = v_hat_Sstar
        w_hat = w_hat_Sstar

    else:
        green_s_star = greens_function('s',force['X0_star'],par)
        green_s      = greens_function('s',force['X0']     ,par)
        green_d      = greens_function('d',force['X0']     ,par)

        [u_hat_Sstar,v_hat_Sstar,w_hat_Sstar] = velocity_FFT_element('Stokeslet',force,green_s_star,par)
        [u_hat_S    ,v_hat_S    ,w_hat_S    ] = velocity_FFT_element('Stokeslet',force,green_s     ,par)
        [u_hat_DP   ,v_hat_DP   ,w_hat_DP   ] = velocity_FFT_element('Dipole'   ,force,green_d     ,par)
        [u_hat_DO   ,v_hat_DO   ,w_hat_DO   ] = velocity_FFT_element('Doublet'  ,force,green_s     ,par)
        [u_hat_Rs   ,v_hat_Rs   ,w_hat_Rs   ] = velocity_FFT_element('Rotlet'   ,force,green_s     ,par)
        [u_hat_Rd   ,v_hat_Rd   ,w_hat_Rd   ] = velocity_FFT_element('Rotlet'   ,force,green_d     ,par)

        h = force['h']
        u_hat = (u_hat_Sstar - u_hat_S) + 2*h*u_hat_DO + h**2*u_hat_DP + 2*h*(u_hat_Rd - u_hat_Rs)
        v_hat = (v_hat_Sstar - v_hat_S) + 2*h*v_hat_DO + h**2*v_hat_DP + 2*h*(v_hat_Rd - v_hat_Rs)
        w_hat = (w_hat_Sstar - w_hat_S) + 2*h*w_hat_DO + h**2*w_hat_DP + 2*h*(w_hat_Rd - w_hat_Rs)

    return (u_hat,v_hat,w_hat)



def velocity_FFT_element(element,force,green,par):

    kk = par['grid']['k']
    mm = par['grid']['m']

    if element == 'Stokeslet':
        F = force['F']
        f1 = F['f1']
        f2 = F['f2']
        f3 = F['f3']
        G_hat       = green['G_hat']
        B_hat       = green['B_hat']
        B_hat_prime = green['B_hat_prime']
        kk2         = green['k2']
        mm2         = green['m2']
        c2          = green['c2']

        u_hat = -f1*(kk2*B_hat + G_hat)   - f2*kk*mm*B_hat           + f3*1j*kk*B_hat_prime
        v_hat = -f1*kk*mm*B_hat           - f2*(mm2*B_hat + G_hat)   + f3*1j*mm*B_hat_prime
        w_hat =  f1*1j*kk*B_hat_prime     + f2*1j*mm*B_hat_prime     + f3*c2*B_hat

    elif element == 'Dipole':
        Q = force['Q']
        q1 = Q['q1']
        q2 = Q['q2']
        q3 = Q['q3']
        Phi_hat     = green['Phi_hat']
        G_hat       = green['G_hat']
        G_hat_prime = green['G_hat_prime']
        kk2         = green['k2']
        mm2         = green['m2']
        c2          = green['c2']

        u_hat =  q1*(kk2*G_hat + Phi_hat) + q2*kk*mm*G_hat           - q3*1j*kk*G_hat_prime
        v_hat =  q1*kk*mm*G_hat           + q2*(mm2*G_hat + Phi_hat) - q3*1j*mm*G_hat_prime
        w_hat = -q1*1j*kk*G_hat_prime     - q2*1j*mm*G_hat_prime     - q3*c2*G_hat

    elif element == 'Doublet':
        Q = force['Q']
        q1 = Q['q1']
        q2 = Q['q2']
        q3 = Q['q3']
        G_hat       = green['G_hat']
        B_hat       = green['B_hat']
        B_hat_prime = green['B_hat_prime']
        kk2         = green['k2']
        mm2         = green['m2']
        c2          = green['c2']

        u_hat = -q1*kk2  *B_hat_prime - q2*kk*mm*B_hat_prime + q3*1j*kk*(c2*B_hat + G_hat)
        v_hat = -q1*kk*mm*B_hat_prime - q2*mm2  *B_hat_prime + q3*1j*mm*(c2*B_hat + G_hat)
        w_hat =  q1*1j*kk*c2*B_hat    + q2*1j*mm*c2*B_hat    + q3*c2*B_hat_prime

    elif element == 'Rotlet':
        F = force['F']
        f1 = F['f1']
        f2 = F['f2']
        f3 = F['f3']
        G_hat       = green['G_hat']
        G_hat_prime = green['G_hat_prime']

        u_hat = -f1*G_hat_prime
        v_hat = -f2*G_hat_prime
        w_hat =  f1*1j*kk*G_hat + f2*1j*mm*G_hat

    # separately take care of the term k^2+m^2=0
    (u00,v00,w00) = zero_term(element,force,green)
    u_hat[0,0,:] = u00
    v_hat[0,0,:] = v00
    w_hat[0,0,:] = w00

    return (u_hat, v_hat, w_hat)



def zero_term(element,force,green):

    if element == 'Stokeslet':
        F = force['F']
        f1 = F['f1']
        f2 = F['f2']
        G0_hat = green['G0_hat']
        u00 = -f1*G0_hat
        v00 = -f2*G0_hat
        w00 = 0
    elif element == 'Dipole':
        Q = force['Q']
        q1 = Q['q1']
        q2 = Q['q2']
        Phi0_hat = green['Phi0_hat']
        u00 = q1*Phi0_hat
        v00 = q2*Phi0_hat
        w00 = 0
    elif element == 'Doublet':
        u00 = 0
        v00 = 0
        w00 = 0
    elif element == 'Rotlet':
        F = force['F']
        f1 = F['f1']
        f2 = F['f2']
        G0_hat_prime = green['G0_hat_prime']
        u00 = -f1*G0_hat_prime
        v00 = -f2*G0_hat_prime
        w00 = 0

    return (u00,v00,w00)



def greens_function(blob_type,X0,par):

    x0 = X0['x']
    y0 = X0['y']
    z0 = X0['z']
    x_a = par['box']['x_a']
    y_a = par['box']['y_a']
    zz = par['grid']['z']
    kk = par['grid']['k']
    mm = par['grid']['m']

    zz1 = zz - z0
    zz2 = zz1**2
    zz3 = zz2*zz1
    zz4 = zz2**2

    kk2 = kk**2
    mm2 = mm**2

    c2 = kk2 + mm2
    # The term c = 0 is taken care of separetely later
    # and can be ignored for now.
    c2[0,0,:] = np.nan #float('NaN')

    # TODO: The following is a dirty trick to avoid that numpy
    # prints a warning when it encounters nan or in case of overflow.
    # An overflow can happen in the argument of exp. However, such 
    # exponentials are multiplied by terms that tend to 0 a lot faster
    # than the exponential can grow. Mathematically everything is fine
    # (use l'Hopital's rule for a limit of the form 0*infinity).
    # But computationally we need to implement this better.
    np.seterr(invalid='ignore')
    np.seterr(over='ignore')
    c1 = sqrt(c2)
    c3 = c2*c1

    splitting = par['splitting']
    if not splitting:
        epsilon = par['reg']['epsilon']
    else:
        epsilon = par['reg']['xi']

    ep0 = 1
    ep1 = epsilon
    ep2 = ep1**2
    ep3 = ep1**3
    ep4 = ep1**4

    delta_hat_x = exp(-0.25*kk2*ep2 - 1j*kk*(x0-x_a)) # NOTE: k has the factor of 2pi/L already included
    delta_hat_y = exp(-0.25*mm2*ep2 - 1j*mm*(y0-y_a)) # NOTE: m has the factor of 2pi/L already included
    delta_hat = delta_hat_x*delta_hat_y

    Exp_Gauss = exp(-zz2/ep2);
    Exp1  = exp(-c1*zz1)
    Exp2  = exp( c1*zz1)
    Erfc1 = erfc(ep1*c1/2 - zz1/ep1)
    Erfc2 = erfc(ep1*c1/2 + zz1/ep1)

    # NOTE: EXP can become numerical infinity for large abs(c).
    # In this case ERFC = 0 and the product ERFC*EXP = 0 as well.
    # In fact, limit c->Infinity of (Exp(c^n)Erfc(c) = 0 for n<2.
    ExpErfc1 = np.zeros(Exp1.shape)
    ExpErfc2 = np.zeros(Exp2.shape)
    Erfc1_pos = (Erfc1>0)
    Erfc2_pos = (Erfc2>0)
    ExpErfc1[Erfc1_pos] = Exp1[Erfc1_pos]*Erfc1[Erfc1_pos]
    ExpErfc2[Erfc2_pos] = Exp2[Erfc2_pos]*Erfc2[Erfc2_pos]

    # NOTE: The formula coeff_erfc = delta_hat/16.*exp(c2*ep2/4)
    # from the paper becomes numerically unstable for large c since
    # the exponential function blows up while delta_hat goes to 0.
    # We use an algebraically simplified expression.
    coeff_erfc = exp(-1j*kk*(x0-x_a) - 1j*mm*(y0-y_a))/16;
    coeff_exp  = delta_hat*0.07052369794346953 # delta_hat*1/(8*sqrt(pi))

    if not splitting:
        if blob_type == 's':
            Phi_hat     = coeff_exp/ep1*( 12 - 8*zz2/ep2  + 2*ep2*c2)*Exp_Gauss
            G_hat       = coeff_erfc/c1*( (-4            )*ExpErfc1 + (-4            )*ExpErfc2 )
            G_hat_prime = coeff_erfc   *( ( 4            )*ExpErfc1 + (-4            )*ExpErfc2 )
            B_hat       = coeff_erfc/c3*( ( 2 + 2*c1*zz1 )*ExpErfc1 + ( 2 - 2*c1*zz1 )*ExpErfc2 )
            B_hat_prime = coeff_erfc/c2*( (   - 2*c1*zz1 )*ExpErfc1 + (   - 2*c1*zz1 )*ExpErfc2 )

            G_hat       += coeff_exp*ep1*( -2         )*Exp_Gauss
            G_hat_prime += coeff_exp*ep0*( 4*zz1/ep1  )*Exp_Gauss
            B_hat       += coeff_exp*ep3*( 2/(ep2*c2) )*Exp_Gauss

        elif blob_type == 'd':
            Phi_hat     = coeff_exp/ep1*( 8 )*Exp_Gauss
            G_hat       = coeff_erfc/c1*( (-4                    )*ExpErfc1 + (-4                    )*ExpErfc2 )
            G_hat_prime = coeff_erfc   *( ( 4                    )*ExpErfc1 + (-4                    )*ExpErfc2 )
            B_hat       = coeff_erfc/c3*( ( 2 - ep2*c2 + 2*c1*zz1)*ExpErfc1 + ( 2 - ep2*c2 - 2*c1*zz1)*ExpErfc2 )
            B_hat_prime = coeff_erfc/c2*( (     ep2*c2 - 2*c1*zz1)*ExpErfc1 + (   - ep2*c2 - 2*c1*zz1)*ExpErfc2 )

            B_hat       += coeff_exp*ep3*( 2/(ep2*c2) )*Exp_Gauss

    else:
        if blob_type == 's':
            c4 = c2**2
            Phi_hat     = coeff_exp/ep1*( 24 - 56*zz2/ep2 + 16*zz4/ep4 - 8*c2*zz2 + 6*ep2*c2 + ep4*c4 )*Exp_Gauss
            G_hat       = coeff_erfc/c1*( (-4            )*ExpErfc1 + (-4            )*ExpErfc2 )
            G_hat_prime = coeff_erfc   *( ( 4            )*ExpErfc1 + (-4            )*ExpErfc2 )
            B_hat       = coeff_erfc/c3*( ( 2 + 2*c1*zz1 )*ExpErfc1 + ( 2 - 2*c1*zz1 )*ExpErfc2 )
            B_hat_prime = coeff_erfc/c2*( (   - 2*c1*zz1 )*ExpErfc1 + (   - 2*c1*zz1 )*ExpErfc2 )

            G_hat       += coeff_exp*ep1*( (-4 -   c2*ep2)         + 4*zz2/ep2 )*Exp_Gauss
            G_hat_prime += coeff_exp*ep0*( (16 + 2*c2*ep2)*zz1/ep1 - 8*zz3/ep3 )*Exp_Gauss
            B_hat       += coeff_exp*ep3*(   1 + 2/(ep2*c2)                    )*Exp_Gauss
            B_hat_prime += coeff_exp*ep2*( (-2           )*zz1/ep1             )*Exp_Gauss

        elif blob_type == 'd':
            Phi_hat     = coeff_exp/ep1*( 16 - 16*zz2/ep2 + 4*ep2*c2)*Exp_Gauss
            G_hat       = coeff_erfc/c1*( (-4                    )*ExpErfc1 + (-4                    )*ExpErfc2 )
            G_hat_prime = coeff_erfc   *( ( 4                    )*ExpErfc1 + (-4                    )*ExpErfc2 )
            B_hat       = coeff_erfc/c3*( ( 2 + ep2*c2 + 2*c1*zz1)*ExpErfc1 + ( 2 + ep2*c2 - 2*c1*zz1)*ExpErfc2 )
            B_hat_prime = coeff_erfc/c2*( (   - ep2*c2 - 2*c1*zz1)*ExpErfc1 + (     ep2*c2 - 2*c1*zz1)*ExpErfc2 )

            G_hat       += coeff_exp*ep1*( -4         )*Exp_Gauss
            G_hat_prime += coeff_exp*ep0*( 8*zz1/ep1  )*Exp_Gauss
            B_hat       += coeff_exp*ep3*( 2/(ep2*c2) )*Exp_Gauss

    (Phi0_hat,G0_hat,G0_hat_prime) = greens_function_zero_term(blob_type,X0,par)
    green = {'Phi_hat':Phi_hat,'G_hat':G_hat,'G_hat_prime':G_hat_prime,
        'B_hat':B_hat,'B_hat_prime':B_hat_prime,'k2':kk2,'m2':mm2,'c2':c2,
        'Phi0_hat':Phi0_hat,'G0_hat':G0_hat,'G0_hat_prime':G0_hat_prime}
    return green



def greens_function_zero_term(blob_type,X0,par):
    z0 = X0['z']

    # consider zero terms only
    zz = par['grid']['z']
    zz_zero = zz[0,0,:]
    zz1 = zz_zero - z0
    zz2 = zz1**2

    splitting = par['splitting']
    if not splitting:
        epsilon = par['reg']['epsilon']
    else:
        epsilon = par['reg']['xi']

    ep1 = epsilon
    ep2 = ep1**2

    coeff_exp  = 0.07052369794346953 # = 1/(8*sqrt(pi))
    Exp = exp(-zz2/ep2)
    Erf = erf( zz1/ep1)

    if not splitting:
        if blob_type == 's':
            Phi0_hat     = coeff_exp/ep1*( 12 - 8*zz2/ep2               )*Exp
            G0_hat       = coeff_exp*ep1*(  2                           )*Exp + 1/2*zz1*Erf
            G0_hat_prime = coeff_exp    *(      4*zz1/ep1               )*Exp + 1/2    *Erf
        elif blob_type == 'd':
            Phi0_hat     = coeff_exp/ep1*(  8                           )*Exp
            G0_hat       = coeff_exp*ep1*(  4                           )*Exp + 1/2*zz1*Erf
            G0_hat_prime =                                                      1/2    *Erf
    else:
        if blob_type == 's':
            zz3 = zz2*zz1
            zz4 = zz2**2
            ep3 = ep1**3
            ep4 = ep1**4
            Phi0_hat     = coeff_exp/ep1*( 24 - 56*zz2/ep2 + 16*zz4/ep4 )*Exp
            G0_hat       = coeff_exp*ep1*(       4*zz2/ep2              )*Exp + 1/2*zz1*Erf
            G0_hat_prime = coeff_exp    *(      16*zz1/ep1 -  8*zz3/ep3 )*Exp + 1/2    *Erf
        if blob_type == 'd':
            Phi0_hat     = coeff_exp/ep1*( 16 - 16*zz2/ep2              )*Exp;
            G0_hat       =                                                      1/2*zz1*Erf
            G0_hat_prime = coeff_exp    *(       8*zz1/ep1              )*Exp + 1/2    *Erf

    return (Phi0_hat,G0_hat,G0_hat_prime)

