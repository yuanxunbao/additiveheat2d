#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:10:53 2020

@author: yigongqin
"""



import numpy as np


def phys_para():    
# NOTE: for numbers entered here, if having units: length in micron, time in second, temperature in K.
    G = 0.02                        # thermal gradient        K/um
    R = 50                          # pulling speed           um/s
    delta = 0.02                    # strength of the surface tension anisotropy         
    k = 0.14                        # interface solute partition coefficient
    c_infm = 1.519                    # shift in melting temperature     K
    Dl = 3000                       # solute diffusion coefficient      um**2/s
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
    
    eps = 1e-4                      #divide-by-zero treatment
    alpha0 = 0                    # misorientation angle in degree
    
    
    lxd = 200                     # horizontal length in micron
    aratio = 2                   # aspect ratio
    nx = 1000                       # number of grids in x   nx*aratio must be int
    dx = lxd/nx/W0
    dt = (dx/W0)**2/(5*Dl_tilde)                   # time step size for forward euler
    Mt = 10
    Tt = 60

    eta = 0.00                    # magnitude of noise
    filename = 'U0.3ds_scn'+'noi' + str('%4.2E'%eta)+'ang'+str(alpha0)+'lx'+ str(lxd)+'nx'+str(nx)+'W'+str('%4.2f'%W0)+'.mat'
    
    return eps, alpha0, lxd, aratio, nx, dt, Mt, eta, filename

def IO_para(W0,lxd):
    
    z0 = lxd/W0*0.1              # initial location of interface in W0
    r0 = 0.5625/W0
    nw = 1                        # number of perturbations Lx= nw*Lambda
    mag = z0*0.05                 # magnitude of sin perturbation  in W0
    nts = 2                      # number of samples in time   Mt/nts must be int
    direc = '.'
    

    return nts, direc, r0

    

    
