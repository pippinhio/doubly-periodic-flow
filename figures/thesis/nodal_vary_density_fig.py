#!/usr/bin/env python

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import params

os.system("mkdir -p " + params.image_path)

#L_x = L_y
#Netflow, n = 10
#data = np.array([
#    [0.8,4.147091e-02,np.nan],
#    [0.9,4.103747e-02,5.045852e-01],
#    [1.0,4.064797e-02,5.030255e-01],
#    [1.2,3.878813e-02,4.754242e-01],
#    [1.5,3.521462e-02,4.052810e-01],
#    [2.0,0.027958312946,2.902447e-01],
#    [3.0,0.0166166795137,1.531182e-01],
#    [4.0,0.0103558880335,9.078309e-02],
#    [5.0,0.00692138163131,5.931988e-02]
#])

#Netflow, n=48
data = np.array([
    [0.8,4.228302e-02,5.080273e-01],
    [0.9,4.179493e-02,5.150534e-01],
    [1.0,4.106869e-02,5.091267e-01],
    [1.2,3.900896e-02,4.740737e-01],
    [1.5,3.485829e-02,3.979689e-01],
    [2.0,2.726186e-02,2.812446e-01],
    [3.0,1.596412e-02,1.468801e-01],
    [4.0,9.893703e-03,8.681120e-02],
    [5.0,6.596606e-03,5.665353e-02]
])


density = data[:,0]**-2
netflow_20_20 = data[:,1]
netflow_30_55 = data[:,2]

fig, ax1 = plt.subplots()
plt.grid()
plt.title('netflow')

ax1.plot(density, netflow_20_20, marker='s', linestyle='--',
         color='blue',markerfacecolor='None',
         label=r'$\theta=20^{\circ}$,$\Psi=20^{\circ}$')
ax1.set_xlabel(r'cilia density')
ax1.set_xlim(0,2.0)
ax1.set_ylabel(r'$\theta=20^{\circ}$,$\Psi=20^{\circ}$', color='blue',fontsize=params.fontsize_latex)
for tl in ax1.get_yticklabels():
    tl.set_color('blue')
ax1.set_ylim(-0.001,0.055)

ax2 = ax1.twinx()
ax2.plot(density, netflow_30_55, marker='o', linestyle='-',
         color='red',
         label=r'$\theta=30^{\circ}$,$\Psi=55^{\circ}$')
ax2.set_ylabel(r'$\theta=30^{\circ}$,$\Psi=55^{\circ}$', color='red',fontsize=params.fontsize_latex)
ax2.set_ylim(-0.01,0.55)
for tl in ax2.get_yticklabels():
    tl.set_color('red')

ax1.legend(loc=3)
ax2.legend(loc=4)

matplotlib.rcParams.update({'font.size': params.fontsize})

plt.savefig('images/vary_density_netflow.pdf')
plt.savefig('images/vary_density_netflow.eps')
plt.show()




fig, ax1 = plt.subplots()
plt.grid()
plt.title('netenergy')

ax1.plot(density, netflow_20_20/density, marker='s', linestyle='--',
         color='blue',markerfacecolor='None',
         label=r'$\theta=20^{\circ}$,$\Psi=20^{\circ}$')
ax1.set_xlabel(r'cilia density')
ax1.set_xlim(0,2.0)
ax1.set_ylabel(r'$\theta=20^{\circ}$,$\Psi=20^{\circ}$', color='blue',fontsize=params.fontsize_latex)
for tl in ax1.get_yticklabels():
    tl.set_color('blue')
ax1.set_ylim(-0.01,0.25)

ax2 = ax1.twinx()
ax2.plot(density, netflow_30_55/density, marker='o', linestyle='-',
         color='red',
         label=r'$\theta=30^{\circ}$,$\Psi=55^{\circ}$')
ax2.set_ylabel(r'$\theta=30^{\circ}$,$\Psi=55^{\circ}$', color='red',fontsize=params.fontsize_latex)
ax2.set_ylim(-0.1,2.5)
for tl in ax2.get_yticklabels():
    tl.set_color('red')

ax1.legend(loc=3)
ax2.legend(loc=1)

matplotlib.rcParams.update({'font.size': params.fontsize})

plt.savefig(params.image_path + 'vary_density_netenergy.pdf')
plt.savefig(params.image_path + 'vary_density_netenergy.eps')
plt.show()

