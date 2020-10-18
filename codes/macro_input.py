

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:02:28 2020

@author: yigongqin, yuanxunbao
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
        
        Q = 230*8  #8.83        # laser input power    
        eta = 0.09      # abosorption coefficient
        rb = 0.5e-3       #radius of heat source
        Vs = 1e-3/2   #scanning speed
        q0 = 2*Q*eta/(pi*rb**2)   # heat intensity        
        
        # environment background 
        
        hc = 5*0#10       # convective heat transfer coefficient
        epsilon = 0.03*0 #0.05  # thermal radiation coeffcient
        sigma = 5.67e-8   # stefan-boltzmann constant
        Te = 25+273       # environmental temperature        
        
        Tcon = Tl
        
    return phys_para


def simu_parameter(rb):  
      
    class simu_para():
        
        # dx = rb/20
    
        lxd = 18e-3
        asp_ratio = 0.5
        nx = 128*4+1
        dx = lxd / (nx-1)
        # nx = 2*nx -1
        #nx = 401
        #dx = lxd/(nx-1)
        
        dt = 1e-3*16
        tend = 3 #100#10 #365
        Mt = int( tend/dt )
        dt = tend/Mt
        
        nts = 1#650
        
        t0 = 0.00#3.2
        ts = 25*dt*2#0.4#Mt*dt-dt#0.7
        
        direc = './'
        filename = 'head2d_temp_nonlinear'+'_nx'+str(nx)+'_asp'+str(asp_ratio)+'_dt'+str('%4.2e'%dt)+'_Mt'+str(Mt)+'.mat'
        # outname = 'macro_output_low'+'_dt'+str('%4.2e'%dt)+'_dx'+str('%4.2e'%dx)+'_Tt'+str('%4.2e'%(dt*Mt))+'.csv'
        
        # outname = 'macro_output_highQ'+'_dt'+str('%4.2e'%dt)+'_dx'+str('%4.2e'%dx)+'_Tt'+str('%4.2e'%(dt*Mt))+'.mat'
        
        # nxs = int((nx-1)/2+1)
        
    return simu_para
        
        

        
        
        