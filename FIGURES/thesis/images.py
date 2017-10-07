#!/usr/bin/env python

from __future__ import division

import numpy as np
from numpy import pi as pi
from numpy import exp, sqrt
from scipy.special import erf

import matplotlib
import matplotlib.pyplot as plt

import params

#element_type = 'singular' 
element_type = 'regular' 
images = True
#images = False #When trying to plot just one Stokeslet using the module
#   stokeslets.py is now the prefered way.


if not images:
    wall = True
#    wall = False

if images:
    ep = 0.2
    x = np.linspace(-1.0,1.0,23,endpoint=True)
    z = np.linspace(-1.0,1.0,23,endpoint=True)
    x0_s = 0.0
    z0_s = 0.5
else:
    ep = 0.2
    x = np.linspace(-1.0,1.0,23,endpoint=True)
    z = np.linspace(-0.0,2.0,23,endpoint=True)
    x0_s = 0.0
    z0_s = 1.0

(xx,zz) = np.meshgrid(x,z)

x0 =  x0_s
z0 = -z0_s
h = z0_s

f1 = 1.0
f3 = 1.0
q1 = -f1
q3 = f3

r_s = sqrt( (xx - x0_s)**2 + (zz - z0_s)**2)
xx_tilde_s = xx - x0_s
zz_tilde_s = zz - z0_s

r = sqrt( (xx - x0)**2 + (zz - z0)**2)
xx_tilde = xx - x0
zz_tilde = zz - z0
qmxm = q1*xx_tilde + q3*zz_tilde

if element_type == 'singular':
    H1_s = 1/(8*pi*r_s**1)
    H2_s = 1/(8*pi*r_s**3)
    H1       =  1/(8*pi*r**1)
    H2       =  1/(8*pi*r**3)
    H1_prime = -1/(8*pi*r**2)
    H2_prime = -3/(8*pi*r**4)
    D1       = -2/(8*pi*r**3)
    D2       =  6/(8*pi*r**5)
if element_type == 'regular':
    H1_s = 1/(4*pi**(3/2)*ep*r_s**0)*( 1)*exp(-r_s**2/ep**2) + 1/(8*pi*r_s**1)*erf(r_s/ep)
    H2_s = 1/(4*pi**(3/2)*ep*r_s**2)*(-1)*exp(-r_s**2/ep**2) + 1/(8*pi*r_s**3)*erf(r_s/ep)
    H1       = 1/(4*pi**(3/2)*ep*r**0)*( 1                )*exp(-r**2/ep**2) + 1/(8*pi*r**1)*erf(r/ep)
    H2       = 1/(4*pi**(3/2)*ep*r**2)*(-1                )*exp(-r**2/ep**2) + 1/(8*pi*r**3)*erf(r/ep)
    H1_prime = 1/(4*pi**(3/2)*ep*r**1)*( 1 - 2*r**2/ep**2 )*exp(-r**2/ep**2) - 1/(8*pi*r**2)*erf(r/ep)
    H2_prime = 1/(4*pi**(3/2)*ep*r**3)*( 3 + 2*r**2/ep**2 )*exp(-r**2/ep**2) - 3/(8*pi*r**4)*erf(r/ep)
    D1       = 1/(4*pi**(3/2)*ep*r**2)*( 2 + 4*r**2/ep**2 )*exp(-r**2/ep**2) - 2/(8*pi*r**3)*erf(r/ep)
    D2       = 1/(4*pi**(3/2)*ep*r**4)*(-6 - 4*r**2/ep**2 )*exp(-r**2/ep**2) + 6/(8*pi*r**5)*erf(r/ep)
    Gs_prime = 1/(4*pi**(3/2)*ep*r**1)*(-2 + 2*r**2/ep**2 )*exp(-r**2/ep**2) + 2/(8*pi*r**2)*erf(r/ep)
    Gd_prime = 1/(4*pi**(3/2)*ep*r**1)*(-2                )*exp(-r**2/ep**2) + 2/(8*pi*r**2)*erf(r/ep)


# Stokeslet
u_St_s = H1_s*f1 + H2_s*(f1*xx_tilde_s + f3*zz_tilde_s)*xx_tilde_s
w_St_s = H1_s*f3 + H2_s*(f1*xx_tilde_s + f3*zz_tilde_s)*zz_tilde_s

# Stokeslet
u_St = -(H1*f1 + H2*(f1*xx_tilde + f3*zz_tilde)*xx_tilde)
w_St = -(H1*f3 + H2*(f1*xx_tilde + f3*zz_tilde)*zz_tilde)

# Doublet
u_Do = 2*h*(H2*(zz_tilde*q1 + xx_tilde*q3) + H2_prime/r*qmxm*xx_tilde*zz_tilde                  )
w_Do = 2*h*(H2*(zz_tilde*q3 + zz_tilde*q3) + H2_prime/r*qmxm*zz_tilde*zz_tilde + H1_prime/r*qmxm)

# Dipole
u_Dp = h**2*(D1*q1 + D2*qmxm*xx_tilde)
w_Dp = h**2*(D1*q3 + D2*qmxm*zz_tilde)


u = u_St_s + u_St + u_Do + u_Dp
w = w_St_s + w_St + w_Do + w_Dp

if element_type == 'regular':
    # Rotlet
    u_Ro1 = 2*h*(-Gs_prime/r)*zz_tilde*q1               
    w_Ro1 = 2*h*(-Gs_prime/r)*(-xx_tilde*q1)

    u_Ro2 = 2*h*( Gd_prime/r)*zz_tilde*q1               
    w_Ro2 = 2*h*( Gd_prime/r)*(-xx_tilde*q1)

    u += u_Ro1 + u_Ro2
    w += w_Ro1 + w_Ro2


if images:
    if element_type == 'singular':
        cutoff = 0.15
        M_s = np.zeros(u.shape, dtype='bool')
        M_s[(xx>(x0_s-cutoff)) & (xx<(x0_s+cutoff)) & (zz>(z0_s-cutoff)) & (zz<(z0_s+cutoff))] = True
        M = np.zeros(u.shape, dtype='bool')
        M[(xx>(x0-cutoff)) & (xx<(x0+cutoff)) & (zz>(z0-cutoff)) & (zz<(z0+cutoff))] = True

        u_St_s = np.ma.masked_array(u_St_s, mask=M_s)
        w_St_s = np.ma.masked_array(w_St_s, mask=M_s)
        u_St   = np.ma.masked_array(u_St, mask=M)
        w_St   = np.ma.masked_array(w_St, mask=M)
        u_Do   = np.ma.masked_array(u_Do, mask=M)
        w_Do   = np.ma.masked_array(w_Do, mask=M)
        u_Dp   = np.ma.masked_array(u_Dp, mask=M)
        w_Dp   = np.ma.masked_array(w_Dp, mask=M)

        fig1,((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

    elif element_type == 'regular':
        fig1,((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')

    ax1.quiver(xx[zz>=0],zz[zz>=0],u_St_s[zz>=0],w_St_s[zz>=0],color='black')
    ax1.quiver(xx[zz<0] ,zz[zz<0] ,u_St_s[zz<0] ,w_St_s[zz<0] ,color='grey')
    ax2.quiver(xx[zz>=0],zz[zz>=0],u_St[zz>=0]  ,w_St[zz>=0]  ,color='black')
    ax2.quiver(xx[zz<0] ,zz[zz<0] ,u_St[zz<0]   ,w_St[zz<0]   ,color='grey')
    ax3.quiver(xx[zz>=0],zz[zz>=0],u_Do[zz>=0]  ,w_Do[zz>=0]  ,color='black')
    ax3.quiver(xx[zz<0] ,zz[zz<0] ,u_Do[zz<0]   ,w_Do[zz<0]   ,color='grey')
    ax4.quiver(xx[zz>=0],zz[zz>=0],u_Dp[zz>=0]  ,w_Dp[zz>=0]  ,color='black')
    ax4.quiver(xx[zz<0] ,zz[zz<0] ,u_Dp[zz<0]   ,w_Dp[zz<0]   ,color='grey')
    ax1.quiver(x0_s,z0_s, f1,     f3,color='red',scale=10,width=0.02)
    ax2.quiver(x0  ,z0  ,-f1,    -f3,color='red',scale=10,width=0.02)
    ax3.quiver(x0  ,z0  , q1,     q3,color='red',scale=10,width=0.02)
    ax3.quiver(x0  ,z0  ,0.0,sqrt(2),color='red',scale=10,width=0.02)
    ax4.quiver(x0  ,z0  , q1,     q3,color='red',scale=10,width=0.02)
    ax1.set_title('Stokeslet')
    ax2.set_title('Stokeslet')
    ax3.set_title('Doublet')
    ax4.set_title('Dipole')
    ax1.axhline(y=0.0,linestyle='dashed',color='black')
    ax2.axhline(y=0.0,linestyle='dashed',color='black')
    ax3.axhline(y=0.0,linestyle='dashed',color='black')
    ax4.axhline(y=0.0,linestyle='dashed',color='black')

    if element_type == 'regular':
        ax5.quiver(xx[zz>=0],zz[zz>=0],u_Ro1[zz>=0],w_Ro1[zz>=0],color='black')
        ax5.quiver(xx[zz<0] ,zz[zz<0] ,u_Ro1[zz<0] ,w_Ro1[zz<0] ,color='grey')
        ax6.quiver(xx[zz>=0],zz[zz>=0],u_Ro2[zz>=0],w_Ro2[zz>=0],color='black')
        ax6.quiver(xx[zz<0] ,zz[zz<0] ,u_Ro2[zz<0] ,w_Ro2[zz<0] ,color='grey')
        ax5.set_title('Rotlet')
        ax6.set_title('Rotlet')
        ax5.axhline(y=0.0,linestyle='dashed',color='black')
        ax6.axhline(y=0.0,linestyle='dashed',color='black')
        ax5.plot(x0,z0,'o',color='red')
        ax6.plot(x0,z0,'o',color='red')

    fig1.text(0.5, 0.04, r'$x$', ha='center', va='center', fontsize=params.fontsize_latex)
    fig1.text(0.03, 0.5, r'$z$', ha='center', va='center', fontsize=params.fontsize_latex, rotation='vertical')
    ax1.xaxis.set_ticks([-0.5,0.0,0.5])
    ax1.set_xticklabels(['-0.5','','0.5'])
    ax1.yaxis.set_ticks([-1.0,-0.5,0.0,0.5,1.0])
    ax1.set_yticklabels([-1.0,'',0.0,'',1.0])
    matplotlib.rcParams.update({'font.size': 12})
    plt.savefig(params.image_path + 'images_%s.eps' % element_type)
    plt.savefig(params.image_path + 'images_%s.pdf' % element_type)
    plt.show()


else:
    fig2 = plt.figure()
    ax0 = fig2.add_subplot(111)
    if wall:
        ax0.quiver(xx[zz>=0],zz[zz>=0],u[zz>=0],w[zz>=0],color='black')
        ax0.quiver(xx[zz<0] ,zz[zz<0] ,u[zz<0] ,w[zz<0] ,color='white')
        ax0.set_title('Stokeslet near a wall')   
    else:
        ax0.quiver(xx[zz>=0],zz[zz>=0],u_St_s[zz>=0],w_St_s[zz>=0],color='black')
        ax0.quiver(xx[zz<0] ,zz[zz<0] ,u_St_s[zz<0] ,w_St_s[zz<0] ,color='white')
        ax0.set_title('Stokeslet in free space')   
    
    ax0.quiver(x0_s,z0_s, f1,f3,color='red',scale=10,width=0.02)
    #ax0.axhline(y=0.0,linestyle='dashed',color='black')
    ax0.set_ylim([-0.1,2.0])
    ax0.set_xlabel(r'$x$',fontsize=params.fontsize_latex)
    ax0.set_ylabel(r'$z$',fontsize=params.fontsize_latex)

    matplotlib.rcParams.update({'font.size': params.fontsize})
    if wall:
        plt.savefig(params.image_path + 'stokeslet_wall_%s.eps' % element_type)
        plt.savefig(params.image_path + 'stokeslet_wall_%s.pdf' % element_type)
    else:
        plt.savefig(params.image_path + 'stokeslet_free_%s.eps' % element_type)
        plt.savefig(params.image_path + 'stokeslet_free_%s.pdf' % element_type)
    plt.show()

# check whether velocity at wall is zero
print(u[zz==0])
print(w[zz==0])

