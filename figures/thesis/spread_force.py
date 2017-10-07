#!/usr/bin/env python

from __future__ import division
import os

import numpy as np
from numpy import pi as pi
from numpy import exp
from scipy.special import erf

import matplotlib
import matplotlib.pyplot as plt

import params

os.system("mkdir -p " + params.image_path)

x = np.linspace( 0.0,1.0,23,endpoint=True)
y = np.linspace( 0.0,1.0,23,endpoint=True)
x0 = 0.5
y0 = 0.5
f1 = 1.0
f2 = 1.0
(xx,yy) = np.meshgrid(x,y)

ep = 0.25
r = np.sqrt( (xx - x0)**2 + (yy - y0)**2)
Phi_eps = 1/(pi**1.5*ep**3)*(2.5-r**2/ep**2)*exp(-r**2/ep**2)

for regularized in {True,False}:
    fig = plt.figure()
    ax0 = fig.add_subplot(111)
    matplotlib.rcParams.update({'font.size': params.fontsize})
    plt.xticks([0.2,0.4,0.6,0.8,1.0],[0.2,0.4,0.6,0.8,1.0])
    ax0.set_xlabel(r'$x$',fontsize=params.fontsize_latex)
    ax0.set_ylabel(r'$y$',fontsize=params.fontsize_latex)

    if regularized:
        ax0.quiver(xx,yy,f1*Phi_eps,f2*Phi_eps,color='red')
        ax0.set_title('Regularized force field')
        plt.savefig(params.image_path + 'spread_force_regular.eps')
        plt.savefig(params.image_path + 'spread_force_regular.pdf')
    else:
        ax0.set_xlim([0,1])
        ax0.set_ylim([0,1])
        ax0.quiver(x0,y0,f1*2,f2*2,color='red',scale=10,width=0.02)
        ax0.set_title('Single point force')
        plt.savefig(params.image_path + 'spread_force_singular.eps')
        plt.savefig(params.image_path + 'spread_force_singular.pdf')

    plt.show()
