#!/usr/bin/env python

from __future__ import division
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import params

os.system("mkdir -p " + params.image_path)

vals = np.array([   1,  2,  3,  4,  5, 10, 20, 40])
diff_plane = np.array([
    2.11544077e-02,
    5.32966252e-03,
    2.02228834e-03,
    9.67308442e-04,
    5.34226687e-04,
    7.77206474e-05,
    1.04788762e-05,
    1.36019226e-06])

diff_no_plane = np.array([
    0.09156219,
    0.05859426,
    0.04255681,
    0.03333227,
    0.02737147,
    0.01441998,
    0.00740012,
    0.00374831])

line_slope_1 = diff_no_plane[0]/(vals**1)
line_slope_3 = diff_plane[0]   /(vals**3)

fig = plt.figure()
ax0 = fig.add_subplot(111)
ax0.loglog(vals,diff_plane   ,params.linestyle1,marker='s',label='with correction')
ax0.loglog(vals,diff_no_plane,params.linestyle2,marker='o',label='without correction')
ax0.loglog(vals,line_slope_1 ,params.linestyle3)
ax0.loglog(vals,line_slope_3 ,params.linestyle3)
ax0.legend(prop={'size':params.fontsize})
ax0.grid(True)
ax0.set_ylim([diff_plane[-1]/10.0,1.0])

#yticks = ax0.yaxis.get_major_ticks()
#yticks[0].label1.set_visible(False)
ax0.set_xlabel(r'$N$',fontsize=params.fontsize_latex)
#ax0.set_ylabel('error')
ax0.set_ylabel('difference between Stokeslet double sum and FFT method')
plt.gcf().subplots_adjust(bottom=0.15) #make room for x-label
#plt.gca().tight_layout()
matplotlib.rcParams.update({'font.size': params.fontsize})
plt.savefig(params.image_path + 'ncopies_R.pdf')
plt.savefig(params.image_path + 'ncopies_R.eps')
plt.show()

