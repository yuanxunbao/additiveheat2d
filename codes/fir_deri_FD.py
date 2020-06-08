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


class gradscalar(object):
    def __init__(self, dx, dy):
    

        self.dx_in = 1.0/dx
        self.dy_in = 1.0/dy       
        

            
    def gradxx(self,v):
        
        return  self.dx_in*( v[1:-1,1:] - v[1:-1,:-1] )
    

    
    def gradzx(self,v):
        
        return  0.25*self.dy_in* ( v[2:,:-1] + v[2:,1:] - v[:-2,:-1] - v[:-2,1:] )


    def gradxz(self,v):
        
        return  0.25*self.dx_in* ( v[:-1,2:] + v[1:,2:] - v[:-1,:-2] - v[1:,:-2] )

   
    def gradzz(self,v):
        
        return  self.dy_in*( v[1:,1:-1] - v[:-1,1:-1] )


    def avgx(self,v):
        
        return  0.5*( v[1:-1,1:] + v[1:-1,:-1] )   
    
    
    def avgz(self,v):
        
        return  0.5*( v[:-1,1:-1] + v[1:,1:-1] )      


    def gradxc(self,v):
        
        return  0.5*self.dx_in* ( v[1:-1,2:] - v[1:-1,:-2] )        

    def gradzc(self,v):
        
        return  0.5*self.dy_in* ( v[2:,1:-1] - v[:-2,1:-1] )   

        
    
class gradflux(object):
    
    def __init__(self, dx, dy):
        
        self.dx_in = 1.0/dx
        self.dy_in = 1.0/dy
        
    def gradx(self, f):     # f is a m+1 *n matrix
        
        return  self.dx_in*( f[:,1:] - f[:,:-1] )
    
    
    def gradz(self, f):     # f is a m* n+1 matrix
        
        return  self.dy_in*( f[1:,:] - f[:-1,:] )  
    
    
    def avx(self,f):
        
        return  0.5*( f[:,1:] + f[:,:-1] )    
    
    def avz(self,f):
        
        return  0.5*( f[1:,:] + f[:-1,:] )    
    
    
    
    
    
    
    
    
    
    
    