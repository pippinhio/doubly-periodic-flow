#!/usr/bin/env python

from __future__ import division
import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import params

os.system("mkdir -p " + params.image_path)

fig1 = plt.figure()
ax = fig1.add_subplot(111,projection='3d')
grid_x = np.linspace(0,1,12,endpoint=True)
grid_y = np.linspace(0,1,12,endpoint=True)
za = 1.2

for i in range(len(grid_x)-1):
    for j in range(len(grid_y)):
        ax.plot([grid_x[i], grid_x[i+1]], [grid_y[j],grid_y[j]],zs=[za,za],color='black')

for i in range(len(grid_x)):
    for j in range(len(grid_y)-1):
        ax.plot([grid_x[i], grid_x[i]], [grid_y[j],grid_y[j+1]],zs=[za,za],color='black')

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

x0 = 0.8
y0 = 0.2
z0 = 0.3
a = Arrow3D([x0,0.75],[y0,0.5],[z0,0.4], mutation_scale=20, lw=1, arrowstyle="-|>", color="red")
ax.add_artist(a)
ax.scatter(x0,y0,z0,color="red",s=20)
ax.plot([0.0,x0],[y0,y0],zs=[0.0,0.0],linestyle='dashed',color='blue')
ax.plot([x0,x0],[0.0,y0],zs=[0.0,0.0],linestyle='dashed',color='blue')
ax.plot([x0,x0],[y0,y0],zs=[0.0,z0],linestyle='dashed',color='blue')
ax.scatter(x0,y0,0.0,color="blue",s=20)

ax.set_xlim([0.0,1.0])
#ax.plot([0.0,1.0],[0.0,0.0],zs=[0.0,0.0],linestyle='solid',color='black')
a = Arrow3D([0.0,1.0],[0.0,0.0],[0.0,0.0], mutation_scale=10, lw=1, arrowstyle="-|>", color="black")
ax.add_artist(a)
ax.set_ylim([0.0,1.0])
#ax.plot([0.0,0.0],[0.0,1.0],zs=[0.0,0.0],linestyle='solid',color='black')
a = Arrow3D([0.0,0.0],[0.0,1.0],[0.0,0.0], mutation_scale=10, lw=1, arrowstyle="-|>", color="black")
ax.add_artist(a)
ax.set_zlim3d([0.0,1.5])
#ax.plot([0.0,0.0],[0.0,0.0],zs=[0.0,1.5],linestyle='solid',color='black')
a = Arrow3D([0.0,0.0],[0.0,0.0],[0.0,1.5], mutation_scale=10, lw=1, arrowstyle="-|>", color="black")
ax.add_artist(a)
ax.axis('off')
ax.scatter(0.0,0.0,0.0,marker='s',color='black')

x2, y2, _ = proj3d.proj_transform(x0+0.03,y0-0.02,z0-0.06, ax.get_proj())
ax.annotate(r'$(x_0,y_0,z_0)$',xy=(x2,y2),xytext=(x2,y2),fontsize=params.fontsize_latex)

ax.scatter(0.05,0.57,za,color="red",s=30)
x2, y2, _ = proj3d.proj_transform(0.05-0.25,0.57-0.03,za, ax.get_proj())
ax.annotate(r'$(x,y,z)$',xy=(x2,y2),xytext=(x2,y2),fontsize=params.fontsize_latex)
x2, y2, _ = proj3d.proj_transform(0.5,0.0,0.0-0.20, ax.get_proj())
ax.annotate(r'$x$',xy=(x2,y2),xytext=(x2,y2),fontsize=params.fontsize_latex)
x2, y2, _ = proj3d.proj_transform(0.0,0.5,0.0+0.15, ax.get_proj())
ax.annotate(r'$y$',xy=(x2,y2),xytext=(x2,y2),fontsize=params.fontsize_latex)
x2, y2, _ = proj3d.proj_transform(0.0-0.08,0.0,0.75, ax.get_proj())
ax.annotate(r'$z$',xy=(x2,y2),xytext=(x2,y2),fontsize=params.fontsize_latex)

plt.savefig(params.image_path + 'schematic_computation.eps')
plt.savefig(params.image_path + 'schematic_computation.pdf')
plt.show()

