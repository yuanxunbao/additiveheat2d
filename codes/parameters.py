#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:41:42 2020

@author: yigongqin
"""
from math import pi
class parameters:
   
    
        #bulk condudtion
        K = 0.007
        rho = 1.0
        Cp = 1.0
        alpha = K/(rho*Cp)
        
        #boundary condition
        #radiation
        Q = 5        #power    
        eta = 1      # abosorption coefficient
        rb = 0.15       #radius of heat source
        Vs = 0.1      #scanning speed
        q0 = 2*Q*eta/(pi*rb**2)   # heat intensity
        
        hc = 0       # convective heat transfer coefficient
        epsilon = 1  # thermal radiation coeffcient
        sigma = 0   # stefan-boltzmann constant
        Te = 0       # environmental temperature
        
        #latent heat
        L = 0.001
        Lm = L/Cp
        Ts = 200
        Tl = 400
     
 