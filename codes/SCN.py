#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:41:05 2020

@author: yigongqin
"""


from math import pi
import numpy as np

class parameters:
    
    G = 140*1e-4   # thermal gradient
    R = 32  # pulling speed
    delta = 0.007 # strength of the surface tension anisotropy
    
    
    k = 0.3   # interface solute partition coefficient
    c_infm = 2 # shift in melting temperature
    Dl = 1000  # solute diffusion coefficient


    
    #T0 = 1  # reference solidus temperature
    #T_m = 1  # melting temperature of pure material A

    alpha = 1.0/(2*np.sqrt(2.0))
    d0 = 1.3e-2  # interface thickness
    #d0 = gamma*T_m/( L*mn*c_inf*(1/k-1) )  # capillary length
    W0 = 108.7*d0
    
    lT = c_infm*( 1.0/k-1 )/G # thermal length
    
    lamd = 5*np.sqrt(2)/8*W0/d0
    tau0 = 0.6267*lamd*W0**2/Dl
    
    
    
    eps = 1e-4
    
    
    # non-dimensionalized parameters
    
    R_tilde = R*tau0/W0
    Dl_tilde = Dl*tau0/W0**2
    lT_tilde = lT/W0