#!/usr/bin/env python

from __future__ import division

import numpy as np
from numpy import pi as pi
from numpy import exp
from scipy.special import erf

import matplotlib
import matplotlib.pyplot as plt

import params

a = 0.0
b = 1.5
N = 1000

eps1 = 0.25
eps2 = 0.3

r = np.linspace(a,b,N,endpoint=True)
H1 = 1/(8*pi*r)
H2 = 1/(8*pi*r**3)

H1_eps1 =  1/(4*pi**1.5*eps1)     *exp(-r**2/eps1**2) + 1/(8*pi*r)   *erf(r/eps1)
H2_eps1 = -1/(4*pi**1.5*eps1*r**2)*exp(-r**2/eps1**2) + 1/(8*pi*r**3)*erf(r/eps1)
H1_eps2 =  1/(4*pi**1.5*eps2)     *exp(-r**2/eps2**2) + 1/(8*pi*r)   *erf(r/eps2)
H2_eps2 = -1/(4*pi**1.5*eps2*r**2)*exp(-r**2/eps2**2) + 1/(8*pi*r**3)*erf(r/eps2)

H1_eps1[0] = 1/(2*eps1   *pi**1.5)
H2_eps1[0] = 1/(6*eps1**3*pi**1.5)
H1_eps2[0] = 1/(2*eps2*pi**1.5)
H2_eps2[0] = 1/(6*eps2**3*pi**1.5)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(r, H1     , params.linestyle1, linewidth=params.linewidth, label=r'$H_1$')
ax1.plot(r, H1_eps1, params.linestyle2, linewidth=params.linewidth, label=r'$H_1^{\Phi_{%g}}$'% eps1)
ax1.plot(r, H1_eps2, params.linestyle3, linewidth=params.linewidth, label=r'$H_1^{\Phi_{%g}}$'% eps2)
ax1.set_xlim([a,b])
ax1.set_ylim([0.0,2.0])
xticks = ax1.xaxis.get_major_ticks()
xticks[0].label1.set_visible(False)
ax1.set_xlabel(r'$r$',fontsize=params.fontsize_latex)
ax1.legend(loc=1,prop={'size':params.fontsize_latex})
ax1.grid(True)
matplotlib.rcParams.update({'font.size': params.fontsize})
plt.savefig(params.image_path + 'H1.eps')
plt.savefig(params.image_path + 'H1.pdf')


fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(r, H2     , params.linestyle1, linewidth=params.linewidth, label=r'$H_2$')
ax2.plot(r, H2_eps1, params.linestyle2, linewidth=params.linewidth, label=r'$H_2^{\Phi_{%g}}$'% eps1)
ax2.plot(r, H2_eps2, params.linestyle3, linewidth=params.linewidth, label=r'$H_2^{\Phi_{%g}}$'% eps2)
ax2.set_xlim([a,b])
ax2.set_ylim([0.0,2.0])
xticks = ax2.xaxis.get_major_ticks()
xticks[0].label1.set_visible(False)
ax2.set_xlabel(r'$r$',fontsize=params.fontsize_latex)
ax2.legend(loc=1,prop={'size':params.fontsize_latex})
ax2.grid(True)
matplotlib.rcParams.update({'font.size': params.fontsize})
plt.savefig(params.image_path + 'H2.eps')
plt.savefig(params.image_path + 'H2.pdf')


#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#ax2.plot(r,H2     ,linestyle='solid' ,color='black',linewidth=2.0,label=r'$H_2$')
#ax2.plot(r,H2_eps1,linestyle='dashed',color='blue' ,linewidth=2.0,label=r'$H_2^{\Phi_{%g}}$'% eps1)
#ax2.plot(r,H2_eps2,linestyle='dotted',color='red'  ,linewidth=2.0,label=r'$H_2^{\Phi_{%g}}$'% eps2)
#ax2.set_xlim([a,b])
#ax2.set_ylim([0.0,2.0])
#ax2.set_xlabel(r'$r$')
#ax2.legend()
#ax2.grid(True)
##ax1.tick_params(axis='x', pad=15)
#plt.xticks([0.25,0.5,0.75,1.0,1.25,1.5],[0.25,0.5,0.75,1.0,1.25,1.5])
#matplotlib.rcParams.update({'font.size': 18})
#plt.savefig('H2.eps')
#plt.savefig('H2.pdf')


plt.show()




