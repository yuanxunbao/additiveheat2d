#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:10:53 2020

@author: yigongqin
"""


from math import pi
import numpy as np

def phy_parameter():
   class phy_para:
    
    # NOTE: for numbers entered here, if having units: length in micron, time in second, temperature in K.
    G = 0.02                  # thermal gradient  
    R = 50                        # pulling speed
    delta = 0.02                 # strength of the surface tension anisotropy        
    k = 0.14                       # interface solute partition coefficient
    c_infm = 1.519                    # shift in melting temperature
    Dl = 3000                     # solute diffusion coefficient
    d0 = 0.02572                   # capillary length -- associated with GT coefficient
    W0 = 0.9375                 # interface thickness  
    
    lT = c_infm*( 1.0/k-1 )/G     # thermal length
    lamd = 5*np.sqrt(2)/8*W0/d0   # coupling constant
    tau0 = 0.6267*lamd*W0**2/Dl   # time scale

    eps = 1e-4                    #divide-by-zero treatment
    # non-dimensionalized parameters based on W0 and tau0
    
    R_tilde = R*tau0/W0
    Dl_tilde = Dl*tau0/W0**2
    lT_tilde = lT/W0
    
   return phy_para

def simu_parameter():   
   class simu_para:
    p = phy_parameter()
    
    alpha0 = 0                    # misorientation angle in degree
    
    
    lx = 230                     # horizontal length in micron
    aratio = 4                   # aspect ratio
    nx = 160                       # number of grids in x   nx*aratio must be int
    dx = lx/nx/p.W0
    dt = (dx/p.W0)**2/(5*p.Dl_tilde)                   # time step size for forward euler
    Mt = 120000
    Tt = 60
    nts = 20                      # number of samples in time   Mt/nts must be int

    z0 = lx/p.W0*0.1              # initial location of interface in W0
    r0 = 0.5625/p.W0
    nw = 1                        # number of perturbations Lx= nw*Lambda
    mag = z0*0.05                 # magnitude of sin perturbation  in W0
    
    eta = 0.00                    # magnitude of noise
    direc = '.'
    filename = 'U0.3ds_scn'+'noi' + str('%4.2E'%eta)+'ang'+str(alpha0)+'lx'+ str(lx)+'nx'+str(nx)+'W'+str('%4.2f'%p.W0)+'.mat'
    
   return simu_para

    

    