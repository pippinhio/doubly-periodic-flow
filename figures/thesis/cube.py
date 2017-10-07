#!/usr/bin/env python

from __future__ import division
import os

import numpy as np
from numpy import random as rd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt

import numpy as np
from itertools import product, combinations

# This is needed to that packages form the parent directory can be
# loaded without using relative imports.
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from src.evaluation import interp
from src.velocity.velocity_from_forces import velocity_from_forces
from src.set_parameters import set_parameters
from src.data_structures import make_forces_struct

import params

os.system("mkdir -p " + params.image_path)

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_aspect("equal")
plt.axis('off')

X0 = np.zeros(1,dtype=[('x',float),('y',float),('z',float)])
X0['x'] = 0.5
X0['y'] = 0.5
X0['z'] = 0.5

F = np.zeros(1,dtype=[('f1',float),('f2',float),('f3',float)])
F['f1'] = 1.0
F['f2'] = 1.0
F['f3'] = 1.0


#draw cube
r = [0, 1]
for s, e in combinations(np.array(list(product(r,r,r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s,e), color="black", linewidth = 2.0)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

##draw coordinate system
#x0 = -0.2
#y0 = -0.3
#le = 0.5
#a = Arrow3D([0.0+x0,0.25+x0],[0.0+y0,0.0+y0],[0.0,0.0], mutation_scale=10, lw=1, arrowstyle="-|>", color="black")
#ax.add_artist(a)
#a = Arrow3D([0.0+x0,0.0+x0],[0.0+y0,0.3+y0],[0.0,0.0], mutation_scale=10, lw=1, arrowstyle="-|>", color="black")
#ax.add_artist(a)
#a = Arrow3D([0.0+x0,0.0+x0],[0.0+y0,0.0+y0],[0.0,0.3], mutation_scale=10, lw=1, arrowstyle="-|>", color="black")
#ax.add_artist(a)
#ax.scatter(x0,y0,0.0,marker='o',color='black')

#ax.annotate(r'$x$',xy=(-0.0420,-0.047),fontsize=params.fontsize_latex)
#ax.annotate(r'$y$',xy=(-0.0439,-0.031),fontsize=params.fontsize_latex)
#ax.annotate(r'$z$',xy=(-0.0572,-0.022),fontsize=params.fontsize_latex)


#draw froce
a = Arrow3D(
    [X0['x'][0],F['f1'][0]*0.75],
    [X0['y'][0],F['f2'][0]*0.75],
    [X0['z'][0],F['f3'][0]*0.75],
    mutation_scale=25, lw=5, arrowstyle="-|>", color="red")
ax.add_artist(a)



#draw planes
ax.plot_surface(
    np.array([[0.0,0.0],[0.0,0.0]]),
    np.array([[0.0,1.0],[0.0,1.0]]),
    np.array([[0.0,0.0],[1.0,1.0]]),
    color = 'red',alpha = 0.3)

ax.plot_surface(
    np.array([[1.0,1.0],[1.0,1.0]]),
    np.array([[0.0,1.0],[0.0,1.0]]),
    np.array([[0.0,0.0],[1.0,1.0]]),
    color = 'red',alpha = 0.3)

ax.plot_surface(
    np.array([[0.0,1.0],[0.0,1.0]]),
    np.array([[0.0,0.0],[0.0,0.0]]),
    np.array([[0.0,0.0],[1.0,1.0]]),
    color = 'grey',alpha = 0.4)

ax.plot_surface(
    np.array([[0.0,1.0],[0.0,1.0]]),
    np.array([[1.0,1.0],[1.0,1.0]]),
    np.array([[0.0,0.0],[1.0,1.0]]),
    color = 'grey',alpha = 0.4)

ax.plot_surface(
    np.array([[0.0,1.0],[0.0,1.0]]),
    np.array([[0.0,0.0],[1.0,1.0]]),
    np.array([[0.0,0.0],[0.0,0.0]]),
    color = 'black',alpha = 1.0)

#ax.scatter(0.5,0.5,0.5,marker='o',color='red',s=50)
ax.annotate(r'$\mathbf{x}_0$',xy=(-0.01,0.00),fontsize=1.25*params.fontsize_latex)
ax.annotate(r'$\mathbf{f}_0$',xy=( 0.025,0.030),fontsize=1.25*params.fontsize_latex,color='red')
#ax.annotate(r'$\mathbf{x}_0$',xy=( 0.000,0.00),fontsize=1.25*params.fontsize_latex)
#ax.annotate(r'$\mathbf{f}_0$',xy=( 0.025,0.022),fontsize=1.25*params.fontsize_latex,color='red')

plt.savefig(params.image_path + 'cube_empty.pdf')
plt.savefig(params.image_path + 'cube_empty.eps')


##draw velocity
#par = set_parameters(domain=[0.0,1.0,0.0,1.0,0.0,1.0],method='FFT',images=True)
#forces = make_forces_struct(X0,F,par)
#sol = velocity_from_forces(forces,par,4)

#x = np.linspace(0.1, 0.9, 6, endpoint = True)
#y = np.linspace(0.1, 0.9, 6, endpoint = True)
#z = np.linspace(0.1, 0.9, 6, endpoint = True)
#xx, yy, zz = np.meshgrid(x,y,z)
#grid = {'x':xx,'y':yy,'z':zz}

#u_interp = interp(sol['u'],grid,par)
#v_interp = interp(sol['v'],grid,par)
#w_interp = interp(sol['w'],grid,par)
#r = np.sqrt(u_interp**2 + v_interp**2 + w_interp**2)
#r = 0.1*r / np.amax(r)

#for i in range(len(x)):
#    for j in range(len(y)):
#        for k in range(len(z)):
#            ax.quiver(
#                xx[i,j,k],
#                yy[i,j,k],
#                zz[i,j,k],
#                u_interp[i,j,k],
#                v_interp[i,j,k],
#                w_interp[i,j,k],
#                length=r[i,j,k]*0 + 0.1,
#                color = 'blue'
#                )



#plt.savefig(params.image_path + 'cube_velo.pdf')
#plt.savefig(params.image_path + 'cube_velo.eps')

plt.show()

