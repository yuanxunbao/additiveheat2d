#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:17:45 2020

@author: yigongqin
"""

# this code is to calculate the first derivatives of a scalar in 2D

# derivatives f = gradx_ix_j (u), x_i is direction, x_j is defined edges
# u defined on grid points, f defined on edges
# u m*n, f (m+1)*n or m*(n+1), expansion at the x_i direction

# u = gradx_i (f) x_i is direction 
# u defined on grid points, f defined on edges
# u m*n, f (m+1)*n or m*(n+1), compression

# boundary conditions are for u
# please note the positive direction
import numpy as np
from numba import njit

pflag = False
            
# @njit(parallel=pflag)
def gradxx(h_in,v):
    
    return  h_in*( v[1:-1,1:] - v[1:-1,:-1] )


# @njit(parallel=pflag)
def gradzx(h_in,v):
    
    return  0.25*h_in* ( v[2:,:-1] + v[2:,1:] - v[:-2,:-1] - v[:-2,1:] )

# @njit(parallel=pflag)
def gradxz(h_in,v):
    
    return  0.25*h_in* ( v[:-1,2:] + v[1:,2:] - v[:-1,:-2] - v[1:,:-2] )

# @njit(parallel=pflag) 
def gradzz(h_in,v):
    
    return  h_in*( v[1:,1:-1] - v[:-1,1:-1] )

# @njit(parallel=pflag)
def avgx(v):
    
    return  0.5*( v[1:-1,1:] + v[1:-1,:-1] )   

# @njit(parallel=pflag)
def avgz(v):
    
    return  0.5*( v[:-1,1:-1] + v[1:,1:-1] )      

# @njit(parallel=pflag)
def gradxc(h_in,v):
    
    return  0.5*h_in* ( v[1:-1,2:] - v[1:-1,:-2] )        

# @njit(parallel=pflag)
def gradzc(h_in,v):
    
    return  0.5*h_in* ( v[2:,1:-1] - v[:-2,1:-1] )   

        
# @njit(parallel=pflag)      
def fgradx(h_in, f):     # f is a m+1 *n matrix
    
    return  h_in*( f[:,1:] - f[:,:-1] )

# @njit(parallel=pflag)
def fgradz(h_in, f):     # f is a m* n+1 matrix
    
    return  h_in*( f[1:,:] - f[:-1,:] )  
    
    
# @njit(parallel=pflag)    
def norm2d(vx,vy):
    return np.sqrt( vx**2 + vy**2 )
    
    
    
    
    
    
    
    
    
    
    
