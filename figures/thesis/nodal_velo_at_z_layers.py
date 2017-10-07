#!/usr/bin/env python

import os
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import params

os.system("mkdir -p " + params.image_path)

directory = 'data/tilt20_cone20/'
disp = pickle.load(open(directory + 'disp_final.p','rb'))

for dim in {'x','y','z'}:
    (nx,ny,nz) = disp[dim].shape
    data = np.reshape(disp[dim],(nx*ny,nz))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('netflow in %s' % (dim))
    ax.set_ylabel('initial height')

    ax.boxplot(data,sym='',vert=False)

    ax.set_yticklabels([0.0,'','','','',0.5,'','','','',1.0,'','','','',1.5,'','','','',2.0])
    ax.xaxis.grid()
    ax.set_xlim(-0.08,0.08)

    # The original color of choice was facecolor='grey', alpha=0.3.
    # However, since transperency doesn't work with eps files, an
    # rgb approximation is used (see mimic_alpha.py by Francesco
    # Montesano).
#    light_grey = np.array([ 0.95019608,  0.95019608,  0.95019608])
#    ax.axhspan(0.0, 11.0, facecolor=light_grey,edgecolor=light_grey)
    ax.axhspan(0.0, 11.0, facecolor='grey',edgecolor='grey',alpha=0.3)

    matplotlib.rcParams.update({'font.size': params.fontsize})
    plt.tight_layout()

    plt.savefig(params.image_path + 'velo_at_z_layers_%s.pdf' % (dim))
    plt.savefig(params.image_path + 'velo_at_z_layers_%s.eps' % (dim))
    plt.show()

