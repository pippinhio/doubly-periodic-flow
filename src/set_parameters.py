#!/usr/bin/env python
"""SETS DEFAULT PARAMETERS (IF NOT SPECIFIED) AND SAVES THEM IN A DICTIONARY
        domain:
        The computational domain is a rectangular box supplied as a list
        domain = [x_a, x_b, y_a, y_b, z_a, z_b].
        If the method of images is used, the wall is always placed at z = 0
        independet of what the computational domain is. In this case, only
        z-values greater or equal to 0 are physical

        dx, dy, dz, z_layers:
        Uniform spacing in x and y is required in order for the FFT to work.
        The spacing in the z direction can be either uniform (numeric dz) or
        non-uniform. If non-uniform spacing is used, the variable z_layers
        sould be supplied as data type numpy.ndarray. For example,
        z_layers = np.array([0.0 0.2 1.0]) creates a grid with three z-layers.

        images:
        If set to True (default), a no-slip boundary condition (zero velocity)
        is enforced at the wall z = 0
        If set to False, the z direction is considered as free (no boundary).

        epsilon:
        Regularization parameter for regularized Stokeslet.
        If epsilon < 4dx then Ewald splitting is recommended.

        method:
        The choices for method are as follows:
        'Real' : No periodicity. Periodic-like behavior can be approximated by
                 using "fake"-copies. The number of such copies in each
                 direction is specified by the variable ncopies_R.
        'FFT'  : Periodicity in x and y. Faster than Ewald but only accurate
                 for epsilon >= 4 max(dx,dy)
        'Ewald': Periodicity in x and y. Accurate for any choice for epsilon
                 > 0.
        All methods can be used either with images or without images.

        xi (only if method == 'Ewald'):
        Gives the size of the splitting parameter. Should be at least as big as
        epsilon. It is recommended to set xi = 4*max(dx,dy).

        r_cutoff (only if method == 'Ewald'):
        Specifies the radius around each force within which the local piece of
        the Ewald splitting is computed.

        ncopies_R (only if method == 'Real'):
        Gives the number of "fake"-copies in each the x and y direction that
        try to mimic periodicity. For instance, if ncopies_R = 5 is used, the
        number of copies is (2*5 + 1)**2 = 121. If free boundary conditions in
        x and y are desired, set ncopies_R = 0.
    """

from __future__ import division

import numpy as np

from src.grid import make_grid


def set_parameters(
    domain=[0.0,1.0,0.0,1.0,0.0,1.0],
    dx=1/64,
    dy=1/64,
    dz=1/64,
    z_layers=None,
    images=True,
    epsilon=None,
    method=None,
    xi=None,
    r_cutoff=None,
    ncopies_R=None
    ):

    if z_layers is None:
        uniform_in_z = True
        spacing = {'dx':dx,'dy':dy,'dz':dz,'uniform_in_z':True}
    else:
        z_layers = np.unique(z_layers)
        uniform_in_z = False
        spacing = {'dx':dx,'dy':dy,'z_layers':z_layers,'uniform_in_z':False}

    x_a = domain[0]
    x_b = domain[1]
    y_a = domain[2]
    y_b = domain[3]
    if uniform_in_z:
        z_a = domain[4]
        z_b = domain[5]
        if z_a == z_b:
            uniform_in_z = False
            spacing.update({'z_layers':z_a,'uniform_in_z':True})
    else:
        z_a = z_layers[0]
        z_b = z_layers[-1]

    L_x = x_b - x_a
    L_y = y_b - y_a
    L_z = z_b - z_a

    box = {'x_a':x_a,'y_a':y_a,'z_a':z_a,
           'x_b':x_b,'y_b':y_b,'z_b':z_b,
           'L_x':L_x,'L_y':L_y,'L_z':L_z}

    grid = make_grid(box,spacing)

    if epsilon is None:
        epsilon  = 4*dx
    # A typical example where epsilon would be a string is
    # epsilon = '4.0dx'.
    if type(epsilon) is str:
        val = float(epsilon[:-2])
        var = epsilon[-2:]
        epsilon = val*spacing[var]

    if method is None:
        if epsilon >= 4*max(dx,dy):
            method = 'FFT'
        else:
            method = 'Ewald'

    if method == 'FFT':
        reg ={'epsilon':epsilon}

    elif method == 'Ewald':
        if xi is None:
            # Set regularization parameter for Ewald splitting.
            # If xi < 4dx => inaccuracies in iFFT.
            # If xi = 4dx => optimal value; yields error ~ 1.0e-16.
            # If xi > 4dx => more (unnecessary) work is spend computing
            #                the local piece.
            xi = 4*max(dx,dy)
        # A typical example where xi would be a string is xi = '4.0dx'.
        if type(xi) is str:
            val = float(xi[:-2])
            var = xi[-2:]
            xi = val*spacing[var]

        if r_cutoff is None:
            # Set cutoff radius for local piece in Ewald splitting.
            # If r_cutoff < 8xi => Local piece doesn't capture enough
            # If r_cutoff = 8xi => optimal value; yields splitting
            #                      error ~ 1.0e-16.
            # If r_cutoff > 8xi => Local piece is chosen too big which
            #                      creates unnecessary work
            r_cutoff = 8*xi
        # A typical example where r_cutoff would be a string is
        # r_cutoff = '8.0xi' or r_cutoff = '32.0dx'.
        if type(r_cutoff) is str:
            val = float(r_cutoff[:-2])
            var = r_cutoff[-2:]
            if var == 'xi':
                r_cutoff = val*xi
            else:
                r_cutoff = xi = val*spacing[var]

        reg ={'epsilon':epsilon,'xi':xi,'r_cutoff':r_cutoff}

    elif method == 'Real':
        if ncopies_R is None:
            # Set the number of "fake"-copies in each direction that
            # try to mimic periodicity.
            ncopies_R = 0
        reg ={'epsilon':epsilon,'ncopies_R':ncopies_R}

    par = {'box':box,'grid':grid,'reg':reg,'method':method,'images':images}
    return par

