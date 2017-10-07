#!/usr/bin/env python

from __future__ import division
import os

import numpy as np
from numpy import random as rd
import matplotlib
import matplotlib.pyplot as plt

# This is needed to that packages form the parent directory can be
# loaded without using relative imports.
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from src.velocity.velocity_from_forces import velocity_from_forces
from src.set_parameters import set_parameters
from src.data_structures import make_forces_struct

import params

os.system("mkdir -p " + params.image_path)

cores = 4

rd.seed(1)
X0 = np.zeros(1,dtype=[('x',float),('y',float),('z',float)])
X0['x'] = 0.5
X0['y'] = 0.5
X0['z'] = 0.5

F = np.zeros(1,dtype=[('f1',float),('f2',float),('f3',float)])
F['f1'] = 1.0
F['f2'] = 0.0
F['f3'] = 1.0

par = set_parameters(domain=[0.0,1.0,0.0,1.0,0.0,2.0],dz=1/64,method='FFT',images=True,epsilon=4/64)
forces = make_forces_struct(X0,F,par)
sol = velocity_from_forces(forces,par,cores)

u_min = np.squeeze(np.apply_over_axes(np.amin, sol['u'], axes=[0,1]))
u_max = np.squeeze(np.apply_over_axes(np.amax, sol['u'], axes=[0,1]))
v_min = np.squeeze(np.apply_over_axes(np.amin, sol['v'], axes=[0,1]))
v_max = np.squeeze(np.apply_over_axes(np.amax, sol['v'], axes=[0,1]))
w_min = np.squeeze(np.apply_over_axes(np.amin, sol['w'], axes=[0,1]))
w_max = np.squeeze(np.apply_over_axes(np.amax, sol['w'], axes=[0,1]))
z = par['grid']['z_vec']

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(u_min,z, params.linestyle1, linewidth=params.linewidth)
ax1.plot(u_max,z, params.linestyle1, linewidth=params.linewidth)
ax1.axvline(x=0.5,linestyle='dashed',linewidth=params.linewidth)
ax1.set_ylabel(r'$z$',fontsize=params.fontsize_latex)
ax1.set_xlabel(r'$u$',fontsize=params.fontsize_latex)
ax1.grid(True)
ax1.set_xlim([-0.5,2.0])
ax1.annotate('maximum velocity', xy=(1.0, 0.6), xytext=(1.2, 0.8),
    arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='left',
    fontsize=params.fontsize)
ax1.annotate('minimum \n velocity', xy=(0.4, 0.6), xytext=(0.2, 0.8),
    arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right',
    fontsize=params.fontsize)
matplotlib.rcParams.update({'font.size': params.fontsize})
plt.savefig(params.image_path + 'flow_at_zlayers_u.eps')
plt.savefig(params.image_path + 'flow_at_zlayers_u.pdf')


fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(v_min,z, params.linestyle1, linewidth=params.linewidth)
ax2.plot(v_max,z, params.linestyle1, linewidth=params.linewidth)
ax2.axvline(x=0.0,linestyle='dashed',linewidth=params.linewidth)
ax2.set_ylabel(r'$z$',fontsize=params.fontsize_latex)
ax2.set_xlabel(r'$v$',fontsize=params.fontsize_latex)
ax2.grid(True)
ax2.set_xlim([-0.5,2.0])
ax2.annotate('maximum \n velocity', xy=(0.1, 0.81), xytext=(0.5, 1.01),
    arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right',
    fontsize=params.fontsize)
ax2.annotate('minimum \n velocity', xy=(-0.1, 0.81), xytext=(-0.48, 1.01),
    arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='left',
    fontsize=params.fontsize)
plt.savefig(params.image_path + 'flow_at_zlayers_v.eps')
plt.savefig(params.image_path + 'flow_at_zlayers_v.pdf')


fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(w_min,z, params.linestyle1, linewidth=params.linewidth)
ax3.plot(w_max,z, params.linestyle1, linewidth=params.linewidth)
ax3.axvline(x=0.0,linestyle='dashed',linewidth=params.linewidth)
ax3.set_ylabel(r'$z$',fontsize=params.fontsize_latex)
ax3.set_xlabel(r'$w$',fontsize=params.fontsize_latex)
ax3.grid(True)
ax3.set_xlim([-0.5,2.0])
ax3.annotate('maximum velocity', xy=(1.0, 0.6), xytext=(1.2, 0.8),
    arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='left',
    fontsize=params.fontsize)
ax3.annotate('minimum \n velocity', xy=(-0.1, 0.6), xytext=(-0.48, 0.8),
    arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='left',
    fontsize=params.fontsize)
plt.savefig(params.image_path + 'flow_at_zlayers_w.eps')
plt.savefig(params.image_path + 'flow_at_zlayers_w.pdf')
plt.show()
