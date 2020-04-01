#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:41:42 2020

@author: yigongqin
"""
from math import pi
class parameters:
   
    
        #bulk condudtion
        K = 0.01
        rho = 1.0
        Cp = 1.0
        alpha = K/(rho*Cp)
        
        #boundary condition
        #radiation
        Q = 3        #power    
        eta = 1      # abosorption coefficient
        rb = 0.2       #radius of heat source
        Vs = 0.075      #scanning speed
        q0 = 2*Q*eta/(pi*rb**2)   # heat intensity
        
        hc = 0.005       # convective heat transfer coefficient
        epsilon = 0.005  # thermal radiation coeffcient
        sigma = 5.67e-8   # stefan-boltzmann constant
        Te = 0       # environmental temperature
        
        #latent heat
        L =  200
        Lm = L/Cp
        Ts = 40
        Tl = 110
     
 