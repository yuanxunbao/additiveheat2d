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
from numpy import shape

class gradscalar(object):
    def __init__(self, dx, dy, BCx, BCy):
    
        self.dx = dx
        self.dy = dy
        self.BCx = BCx    # x BC of u
        self.BCy = BCy
       
        
    def add_BCs(self,u):
        
        m,n = u.shape
        ub = np.zeros((m+2,n+2))
        
        ub[1:-1,1:-1] = u
        
        if self.BCx == 'P':
                     
            ub[:,0] = ub[:,-2] 
            ub[:,-1] = ub[:,1]
            
            
        if self.BCy == 'P':
            
            ub[0,:] = ub[-2,:]
            ub[-1,:] = ub[1,:]
                
       
        if self.BCx == 'R':  # here just no flux, make the code more general
            
            ub[:,0] = ub[:,2]
            ub[:,-1] = ub[:,-3]        
            
        if self.BCy == 'R':
            
            ub[0,:] = ub[2,:]
            ub[-1,:] = ub[-3,:]
        
        
        #print(ub)
        return ub
            
    def gradxx(self,u):
        
        v = self.add_BCs(u)
        return ( v[1:-1,1:] - v[1:-1,:-1] )/self.dx
    

    
    def gradzx(self,u):
        
        v = self.add_BCs(u)
        return ( v[2:,:-1] + v[2:,1:] - v[:-2,:-1] - v[:-2,1:] )/(4*self.dy)


    def gradxz(self,u):
        
        v = self.add_BCs(u)
        return ( v[:-1,2:] + v[1:,2:] - v[:-1,:-2] - v[1:,:-2] )/(4*self.dx)


   
    def gradzz(self,u):
        
        v = self.add_BCs(u)
        return ( v[1:,1:-1] - v[:-1,1:-1] )/self.dy


    def avgx(self,u):
        
        v = self.add_BCs(u)
        return ( v[1:-1,1:] + v[1:-1,:-1] )/2    
    
    
    def avgz(self,u):
        
        v = self.add_BCs(u)
        return ( v[:-1,1:-1] + v[1:,1:-1] )/2      


    def gradxc(self,u):
        
        v = self.add_BCs(u)
        return ( v[1:-1,2:] - v[1:-1,:-2] )/(2*self.dx)        

    def gradzc(self,u):
        
        v = self.add_BCs(u)
        return ( v[2:,1:-1] - v[:-2,1:-1] )/(2*self.dy)   

        
    
class gradflux(object):
    
    def __init__(self, dx, dy):
        
        self.dx = dx
        self.dy = dy
        
    def gradx(self, f):     # f is a m+1 *n matrix
        
        return ( f[:,1:] - f[:,:-1] )/self.dx
    
    
    def gradz(self, f):     # f is a m* n+1 matrix
        
        return ( f[1:,:] - f[:-1,:] )/self.dy  
    
    
    def avx(self,f):
        
        return ( f[:,1:] + f[:,:-1] )/2    
    
    def avz(self,f):
        
        return ( f[1:,:] + f[:-1,:] )/2    
    
    
    
    
    
    
    
    
    
    
    