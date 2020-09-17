#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:41:42 2020

@author: yigongqin
"""
from math import pi

def phys_parameter():
    class phys_para:
   
    
        # Properties of Al-Cu
        k = 0.14
        Tm = 933.3
        c_infty = 3
        liq_slope = 2.6
        c_infm = c_infty*liq_slope
        Ts = Tm - c_infm/k
        Tl = Tm - c_infm
        K = 210                 # heat conductivity J/( m*s*K )
        rho = 2768              # density  kg/m^3
        Cp = 900                # specific heat
        alpha = K/(rho*Cp)

        L =  3.95e5                # latent heat
        Lm = L/Cp      
        
        # laser welding condition
        
        Q = 160*30        # laser input power    
        eta = 0.09      # abosorption coefficient
        rb = 75e-6*30       #radius of heat source
        Vs = 1e-3        #scanning speed
        q0 = 2*Q*eta/(pi*rb**2)   # heat intensity        
        
        # environment background 
        
        hc = 10       # convective heat transfer coefficient
        epsilon = 0.05  # thermal radiation coeffcient
        sigma = 5.67e-8   # stefan-boltzmann constant
        Te = 25+273       # environmental temperature        
        
        Tcon = Tl
        
    return phys_para


def simu_parameter(rb):  
      
    class simu_para():
        
        dx = rb/10
    
        lxd = 800e-6*30 
        asp_ratio = 0.5
        nx = int(lxd/dx)
        nx = nx if nx%2 == 1 else nx+1
        dx = lxd/nx
        nx = 2*nx -1
        #nx = 401
        #dx = lxd/(nx-1)
        dt = 0.01
        Mt = 650#10 #365
        nts = 1#650
        
        t0 = 3.2
        
        direc = '.'
        filename = 'Tsheat2d' + '_lx'+ str(lxd)+'_nx'+str(nx)+'_asp'+str(asp_ratio)+'_dt'+str('%4.2e'%dt)+'_Mt'+str(Mt)+'.mat'
        outname = 'macro_output_Ts'+'_t0'+str('%4.2e'%t0)+'_dt'+str('%4.2e'%dt)+'_dx'+str('%4.2e'%dx)+'_Tt'+str('%4.2e'%(dt*Mt))+'.csv'
        
        nxs = int((nx-1)/2+1)
        
    return simu_para
        
        

        
        
        
        
        
        