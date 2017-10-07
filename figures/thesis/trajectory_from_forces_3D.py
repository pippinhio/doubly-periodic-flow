#!/usr/bin/env python

from __future__ import division
import os
import pickle

import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# This is needed to that packages form the parent directory can be
# loaded without using relative imports.
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from src.grid import ndgrid
from src.evaluation import interp, map_into_main_domain
from src.ode_solve.ode_solve import adams_bashforth
from src.velocity.velocity_from_forces_exact import velocity_from_forces_exact
from src.data_structures import make_forces_struct

import params 

cores = 1

directory = 'data/tilt30_cone55/'

fig = plt.figure()
#fig, ax = plt.subplots()
ax = fig.gca(projection='3d')

par = pickle.load(open(directory + 'par.p','rb'))
x_a = par['box']['x_a']
x_b = par['box']['x_b']
y_a = par['box']['y_a']
y_b = par['box']['y_b']

npart = 1
part_X = np.zeros(npart,dtype=[('x',float),('y',float),('z',float)])
part_X['x'] = np.array([0.0])
part_X['y'] = np.array([0.0])
part_X['z'] = np.array([1.3])
trajectory = [{'x':[],'y':[],'z':[]} for i in range(npart)]
x_plot = []
y_plot = []

part_U_old = {'u':{},'v':{},'w':{}}

ncycles = 3
cilia_config = pickle.load(open(directory + 'cilia_config.p','rb'))
nsteps = cilia_config['nsteps']
dt = cilia_config['dt']

count = 0
plot_nr = 0
# Adams-Bashforth is implemented up to order 4.  In order to
# start the method lower order AB-methods are used.
AB_order = 1
for cycle in range(ncycles):
    for step in ((np.arange(0,nsteps) - 8) % nsteps):#range(64):#range(nsteps):
        print 'step %g of %g ' % (count+1,nsteps*ncycles)
        forces_vec = np.load(directory + 'forces%04d.npz' % step)
        forces = make_forces_struct(forces_vec['X0'],forces_vec['F'],par)
        X0 = forces_vec['X0']
        forces_vec.close()
        U = velocity_from_forces_exact(forces,part_X,par,cores)

        part_U_old['u'].update({'n':U['u']})
        part_U_old['v'].update({'n':U['v']})
        part_U_old['w'].update({'n':U['w']})
        (part_X,part_U_old) = adams_bashforth(part_X,part_U_old,dt,AB_order)
        if (AB_order < 4):
            AB_order += 1
        count += 1

        for i in range(npart):
            trajectory[i]['x'].append(part_X['x'][i])
            trajectory[i]['y'].append(part_X['y'][i])
            trajectory[i]['z'].append(part_X['z'][i])

        if step == 0:
            plot_nr += 1
        if 1 <= plot_nr <= nsteps:
            if step == 64-8:
                ax.plot(
                    X0['x'][2:],
                    X0['y'][2:],
                    X0['z'][2:],
                    linestyle='solid',
                    color='black',
                    linewidth=6,
                    alpha=1.0)
            else:
                ax.plot(
                    X0['x'],
                    X0['y'],
                    X0['z'],
                    linestyle='solid',
                    color='blue',
                    linewidth=4,
                    alpha=abs(2.0*step/nsteps - 1))

#plot particle trajectory
for i in range(npart):
    ax.plot(
        trajectory[i]['x'],
        trajectory[i]['y'],
        trajectory[i]['z'],
        linestyle='solid',color='r')


#plot particle
ax.scatter(
    trajectory[0]['x'][-1],
    trajectory[0]['y'][-1],
    trajectory[0]['z'][-1],
    marker='o',
    color='red',
    alpha = 1.0
    )

#plot surface
ax.plot_surface(
    np.array([[-1.0, 1.0],[-1.0, 1.0]]),
    np.array([[-1.0,-1.0],[ 1.0, 1.0]]),    
    np.array([[ 0.0, 0.0],[ 0.0, 0.0]]),
    color = 'grey',alpha = 0.2)


#plot arrows
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

a = Arrow3D(
    [trajectory[0]['x'][0],trajectory[0]['x'][-1]],
    [trajectory[0]['y'][0]+0.5,trajectory[0]['y'][-1]+0.5],
    [trajectory[0]['z'][0],trajectory[0]['z'][-1]],
    mutation_scale=20, lw=5, arrowstyle="-|>", color="red")
ax.add_artist(a)

t_circ = np.linspace(1.75*pi,2.0*pi,100,endpoint=True)
x_circ = 0.5*cos(t_circ)
y_circ = 0.5*sin(t_circ)
z_circ = np.linspace(0.0,0.0,100,endpoint=True)
ax.plot(x_circ,y_circ,z_circ,linestyle='solid',linewidth=3,color='blue')

a = Arrow3D(
    [x_circ[-1],x_circ[-1]],
    [y_circ[-1],y_circ[-1]+0.15],
    [z_circ[-1],z_circ[-1]],
    mutation_scale=20, lw=5, arrowstyle="-|>", color="blue")
ax.add_artist(a)


#plot coordinate system
x0 = -0.9
y0 = -0.9
le = 0.5
a = Arrow3D([0.0+x0,0.6+x0],[0.0+y0,0.0+y0],[0.0,0.0], mutation_scale=10, lw=1, arrowstyle="-|>", color="black")
ax.add_artist(a)
a = Arrow3D([0.0+x0,0.0+x0],[0.0+y0,0.33+y0],[0.0,0.0], mutation_scale=10, lw=1, arrowstyle="-|>", color="black")
ax.add_artist(a)
a = Arrow3D([0.0+x0,0.0+x0],[0.0+y0,0.0+y0],[0.0,0.5], mutation_scale=10, lw=1, arrowstyle="-|>", color="black")
ax.add_artist(a)
ax.scatter(x0,y0,0.0,marker='o',color='black')

ax.annotate(r'$x$',xy=(-0.0490,-0.047),fontsize=params.fontsize_latex)
ax.annotate(r'$y$',xy=(-0.0400,-0.026),fontsize=params.fontsize_latex)
ax.annotate(r'$z$',xy=(-0.0577,-0.007),fontsize=params.fontsize_latex)


#set up plot
ax.set_xlim([x_a,x_b])
ax.set_ylim([y_a,y_b])
ax.set_zlim([0.0,2.0])
ax.view_init(30, 340)
plt.axis('off')
plt.draw()
plt.savefig(params.image_path +  + 'nodal_fancy.pdf')
plt.savefig(params.image_path +  + 'nodal_fancy.eps')
plt.show()



