#!/usr/bin/env python

from __future__ import division

import matplotlib
import matplotlib.pyplot as plt

import params

eps_over_dx = [0.1, 0.5, 1.0, 1.5, 2.0, 2.2, 3.0, 4.0]
diff_Real_FFT = [
    1.40897807e+00,
    9.08025259e-01,
    8.70029822e-02,
    2.15102372e-03,
    1.87374220e-05,
    3.22068087e-06,
    3.93577527e-06,
    4.67971373e-06
    ]

diff_Real_Ewald = [  
    6.13139046e-07,
    1.17428635e-06,
    1.78016977e-06,
    2.42291133e-06,
    3.00290234e-06,
    3.22068087e-06,
    3.93577527e-06,
    4.67971373e-06
    ]

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.semilogy(eps_over_dx,diff_Real_FFT,params.linestyle2,marker='o',label='method without splitting')
#ax.semilogy(eps_over_dx,diff_Real_Ewald,params.linestyle1,marker='s',label='method with Ewald splitting')
ax.semilogy(eps_over_dx,diff_Real_Ewald,params.linestyle1,marker='s',label='Ewald vs. Stokeslet double sum')
ax.semilogy(eps_over_dx,diff_Real_FFT,params.linestyle2,marker='o',label='FFT vs. Stokeslet double sum')

ax.legend(loc=1)
ax.grid(True)
#ax.set_xlim([0.0,4.0])
ax.set_ylim([10.0**-07,10.0**1])
ax.set_xlim([0.0,6.0])
#ax.set_ylim([10.0**-16,10.0])
ax.set_xlabel(r'$\varepsilon / \Delta x$',fontsize=params.fontsize_latex)
ax.set_ylabel('relative error in max norm',fontsize=params.fontsize)
#plt.gcf().subplots_adjust(bottom=0.15) #make room for x-label
matplotlib.rcParams.update({'font.size': params.fontsize})
plt.savefig(params.image_path + 'FFT_Ewald_Real.pdf')
plt.savefig(params.image_path + 'FFT_Ewald_Real.eps')
plt.show()
