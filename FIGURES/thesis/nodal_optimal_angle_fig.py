#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import params

theta = np.array([0,5,10,15,20,25,30,35,40])

# 10 points
#netflow = np.array([
## psi =         45,           55,           60
#    [-1.613293e-16, 1.526557e-16,-1.890849e-16], # theta = 0
#    [ 4.698076e-02, 6.907488e-02, 8.340080e-02], # theta = 5
#    [ 9.453007e-02, 1.397179e-01, 1.696953e-01], # theta = 10
#    [ 1.428267e-01, 2.141752e-01, 2.633233e-01], # theta = 15
#    [ 1.927770e-01, 2.954444e-01, 3.698817e-01], # theta = 20
#    [ 2.463611e-01, 3.895180e-01, 4.974959e-01], # theta = 25
#    [ 3.060341e-01, 5.030255e-01,       np.nan], # theta = 30
#    [ 3.764885e-01,       np.nan,       np.nan], # theta = 35
#    [ 4.642784e-01,       np.nan,       np.nan]  # theta = 40
#])

# 48 points
netflow = np.array([
# psi =         45,           55,           60
    [-4.658253e-14,-4.726775e-14,-5.113097e-14], # theta = 0
    [ 4.781785e-02, 7.045953e-02, 8.517761e-02], # theta = 5
    [ 9.600876e-02, 1.425405e-01, 1.734167e-01], # theta = 10
    [ 1.451607e-01, 2.184160e-01, 2.687103e-01], # theta = 15
    [ 1.964275e-01, 3.013916e-01, 3.767140e-01], # theta = 20
    [ 2.514507e-01, 3.965454e-01, 5.034056e-01], # theta = 25
    [ 3.119144e-01, 5.091267e-01,       np.nan], # theta = 30
    [ 3.827137e-01,       np.nan,       np.nan], # theta = 35
    [ 4.693416e-01,       np.nan,       np.nan]  # theta = 40
])

netflow45 = netflow[:,0]
netflow55 = netflow[:,1]
netflow60 = netflow[:,2]

fig = plt.figure()
ax = fig.gca()
ax.set_xlabel(r'$\psi + \theta$',fontsize=params.fontsize_latex)
ax.set_ylabel('netflow')
ax.set_xlim(40,90)
ax.set_ylim(0,0.55)
plt.plot(theta + 60, netflow60, marker='s', linestyle='--',
         color='blue',label=r'$\Psi=60^{\circ}$',markerfacecolor='None')
plt.plot(theta + 55, netflow55, marker='o', linestyle='-',
         color='red',label=r'$\Psi=55^{\circ}$')
plt.plot(theta + 45, netflow45, marker='d', linestyle='-.',
         color='black',label=r'$\Psi=45^{\circ}$',markerfacecolor='None')

plt.legend(loc=2)
plt.grid()
matplotlib.rcParams.update({'font.size': params.fontsize})
plt.tight_layout()

plt.savefig(params.image_path + 'optimal_angle.pdf')
plt.savefig(params.image_path + 'optimal_angle.eps')
plt.show()
