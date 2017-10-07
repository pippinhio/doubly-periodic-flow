#!/usr/bin/env python

from __future__ import division

import numpy as np
from numpy import pi
from numpy import exp
from scipy.special import erf

import matplotlib
import matplotlib.pyplot as plt

# This is needed to that packages form the parent directory can be
# loaded without using relative imports.
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from src.cilia.planar import parametrize
from src.set_parameters import set_parameters

import params

fig = plt.figure()
ax = fig.add_subplot(111)

nsteps = 13

cilia_config = {
    'T':1.0,
    'npoints':1000,
    'ncilia_x':1,
    'ncilia_y':1,    
    'nsteps':nsteps,
    'dt':1.0/13,
    'beat_pattern':'SandersonSleigh',
    'L':1.0,
    'ncilia':1
}

par = set_parameters([-1.0,1.0,-1.0,1.0,0.0,1.0])


for i in range(nsteps)[::-1]:
    X0, U = parametrize(i/nsteps,cilia_config,par)
    if i < 5:
        plt.plot(X0['x'],X0['z'],linewidth=4,color='red',alpha=1-(i+2)/(nsteps+2))
    else:
        plt.plot(X0['x'],X0['z'],linewidth=4,color='blue',alpha=1-(i+2)/(nsteps+2))
    plt.annotate(i+1,xy=(X0['x'][0],X0['z'][0]))

ax.arrow(-0.5,1.6, 0.5, 0.0, head_width=0.05, head_length=0.07, facecolor='red', edgecolor='red', linewidth=5.0)
ax.annotate('effective stroke',xy=(0.2,1.575),color='red',fontsize=params.fontsize)
ax.arrow(-0.05,1.4,-0.3, 0.0, head_width=0.05, head_length=0.07, facecolor='blue', edgecolor='blue', linewidth=4.0, alpha = 1.0)
ax.annotate('recovery stroke',xy=( 0.2,1.375),color='blue',fontsize=params.fontsize)

ax.set_xlim([-1,1])
ax.set_ylim([0,2])
ax.plot([-1.0,1.0],[0.0,0.0],linewidth=5,color='black')
ax.arrow(-0.9,0.1, 0.0, 0.2, head_width=0.03, head_length=0.03, facecolor='black', edgecolor='black', linewidth=0.5)
ax.arrow(-0.9,0.1, 0.15, 0.0, head_width=0.03, head_length=0.03, facecolor='black', edgecolor='black', linewidth=0.5)
ax.annotate(r'$x$',xy=(-0.71,0.085),fontsize=params.fontsize_latex)
ax.annotate(r'$z$',xy=(-0.92,0.36),fontsize=params.fontsize_latex)
ax.plot(-0.9,0.1,marker='o',color='black')

plt.axis('off')
plt.savefig(params.image_path + 'pulmonary_cilium.pdf')
plt.savefig(params.image_path + 'pulmonary_cilium.eps')
plt.show()
