#!/usr/bin/env python

from __future__ import division
import os

import numpy as np
from numpy import pi
from numpy import exp
from scipy.special import erf

import matplotlib
import matplotlib.pyplot as plt

import params

os.system("mkdir -p " + params.image_path)

a = -1.0
b =  1.0
N = 1000

eps1 = 0.25
eps2 = 0.3

r = np.linspace(a,b,N,endpoint=True)

Phi_eps1 = 1/(pi**1.5*eps1**3)*(2.5-r**2/eps1**2)*exp(-r**2/eps1**2)
Phi_eps2 = 1/(pi**1.5*eps2**3)*(2.5-r**2/eps1**2)*exp(-r**2/eps2**2)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(r, Phi_eps1, params.linestyle2, linewidth=params.linewidth, label=r'$\Phi_{%g}$'% eps1)
ax1.plot(r, Phi_eps2, params.linestyle3, linewidth=params.linewidth, label=r'$\Phi_{%g}$'% eps2)
ax1.set_xlim([a,b])
yticks = ax1.yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
ax1.set_xlabel(r'$r$',fontsize=params.fontsize_latex)
ax1.legend(loc=1,prop={'size':params.fontsize_latex})
ax1.grid(True)
matplotlib.rcParams.update({'font.size': params.fontsize})
plt.savefig(params.image_path + 'blobs.eps')
plt.savefig(params.image_path + 'blobs.pdf')
plt.show()

