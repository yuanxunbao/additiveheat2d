#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:10:53 2020

@author: yigongqin
"""



import numpy as np
from math import pi

def phys_para():    
# NOTE: for numbers entered here, if having units: length in micron, time in second, temperature in K.
    G = 0.02                        # thermal gradient        K/um
    R = 50                          # pulling speed           um/s
    delta = 0.02                    # strength of the surface tension anisotropy         
    k = 0.14                        # interface solute partition coefficient
    c_infm = 1.519                  # shift in melting temperature     K
    Dl = 3000                       # liquid diffusion coefficient      um**2/s
    d0 = 0.02572                    # capillary length -- associated with GT coefficient   um
    W0 = 0.9375                     # interface thickness      um
    
    lT = c_infm*( 1.0/k-1 )/G       # thermal length           um
    lamd = 5*np.sqrt(2)/8*W0/d0     # coupling constant
    tau0 = 0.6267*lamd*W0**2/Dl     # time scale               s
    
    
    # non-dimensionalized parameters based on W0 and tau0
    
    R_tilde = R*tau0/W0
    Dl_tilde = Dl*tau0/W0**2
    lT_tilde = lT/W0

    return delta, k, lamd, R_tilde, Dl_tilde, lT_tilde, W0, tau0


def simu_para(W0,Dl_tilde):
    cut = 1e-6    
    eps = 1e-8                      #divide-by-zero treatment
    alpha0 = 0                    # misorientation angle in degree
    
    fc = 1
    lxd = 350                     # horizontal length in micron
    aratio = 1                  # aspect ratio
    nx = 250*fc                      # number of grids in x   nx*aratio must be int
    dx = lxd/nx/W0
    dt = (dx)**2/(5*Dl_tilde)                   # time step size for forward euler
    Mt = 50000*fc**2                                 # number of time steps
    Tt = 60                                    # total time

    eta = 0.02                   # magnitude of noise
    filename = '1noi' + str('%4.2E'%eta)+'ang'+str(alpha0)+'lx'+ str(lxd)+'nx'+str(nx)+'AR'+str(aratio)+'.mat'
    nxs = int(nx/fc)
	    
    return eps, alpha0, lxd, aratio, nx, dt, Mt, eta, filename, nxs

def IO_para(W0,lxd):
    
    U_0 = - 0.3                  # initial value for U, -1< U_0 < 0
    seed = 1                     # randnom seed number
    nts = 100                      # number of snapshots in time   Mt/nts must be int
    direc = '/work/07428/ygqin/frontera/data'                  # saving directory
    
    return  U_0, seed, nts, direc


def seed_initial(xx,lx,zz): 
    
    r0 = 0.5625
    r = np.sqrt( (xx-lx/2) **2+(zz)**2)     
    psi0 = r0 - r 
    
    return psi0


def planar_initial(lx,zz):
    
    z0 = lx*0.05                   # initial location of interface in W0   
    psi0 = z0 - zz
    
    return psi0


def sins_initial(lx,nx,xx,zz): 
    
    k_max = int(np.floor(nx/10))    # max wavenumber, 12 grid points to resolve the highest wavemode
    
    amp = 1
    A = (np.random.rand(k_max)-0.5) * amp  # amplitude, draw from [-1,1] * eta
    x_c = np.random.rand(k_max)*lx;                  # shift, draw from [0,Lx]    
    z0 = lx*0.1;                               # level of z-axis
    
    # sinusoidal perturbation
    sp = 0*zz
    for kk in range(k_max):
       
        sp = sp + A[kk]*np.sin(2*pi*kk/lx* (xx-x_c[kk]) );
        
    psi0 = -(zz-z0-sp)
    
    return psi0
    

    










