#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:45:21 2020

@author: yigongqin
"""

import numpy as np
from parameters import parameters 
from scipy import sparse as sp
from scipy.sparse import linalg as la
import matplotlib.pyplot as plt
from scipy.io import savemat as save
#from scipy.sparse import csc_matrix

def radiation(t,h):
    #print('nx=',nx,'amazing!')
    return p.q0 * np.exp( -2*( np.arange(nx).reshape((nx,1))*h - p.Vs*t )**2/p.rb**2 )
    

def Up_boundary(t,h,Tu):
    
    qs = radiation(t,h)
    
    return -2*h/p.K*( -qs + p.hc*(Tu-p.Te) + p.epsilon*p.sigma*(Tu**4-p.Te**4) )

def imex_latent(y):
    
    down = y>p.Ts
    up = y<p.Tl
    
    y_l = y*down*up
    ind = np.ones((nv,1))*down*up
    ind = np.reshape(ind,(nv))
    A_l = sp.diags(ind,format='csc')
    
    return y_l,A_l



#====================parameters==========================

p = parameters()

N = 500
M = 100
nx= N+1  #x grids
ny= M+1  #y grids
nv= ny*nx #number of variables

h = 0.02  #mesh width

dt= 0.1  #length of time step
Mt= 100  #time steps
t=0

C = p.alpha*dt/h**2  #CFL number

lat = p.Lm/dt/(p.Tl-p.Ts)


#save('para.mat',{'nx': nx})

#=====================Generate matrix/solver========================

I = sp.eye(nx*ny,format='csc')

Dxx = sp.diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx)).toarray()
Dxx[0,1]=2; Dxx[-1,-2]=2 #Riemann boundary condition 

Dyy = sp.diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny)).toarray()
Dyy[0,1]=2; Dyy[-1,-2]=2


L = sp.kronsum(Dyy,Dxx,format='csc')

A0 = I - C*L
#Aarr = A.toarray()

#pd =is_pos_def(A.toarray())   #check if A matrix is positive definite 
# LU decomposition Pr*A*Pc = LU
lu = la.splu(A0)

LU_x = lambda x: lu.solve(x)
pvec = la.LinearOperator((nv, nv), LU_x)

##====================Initial condition=========================
Temp = np.zeros((nv,6))

#T = np.zeros((ny,nx))      # temperature T0
#T0 = np.arange(nv).reshape((ny,nx))
T0 = 1*np.ones((ny,nx))
y = np.reshape(T0,(nv,1),order='F')  #T0/y0
Temp[:,[0]] = y

y_l,A_l=imex_latent(y)


#plt.imshow(T0,cmap=plt.get_cmap('hot'))

Tu = y[0:-1:ny]
U = Up_boundary(t,h,Tu)          # U0

bU = np.zeros((nv,1))
bU[0:-1:ny] = U

#======================time evolusion=======================

for i in range(1000):
    
    # obtain right hand side
    b = y + lat*y_l + C*bU            # b_n-1 from T_n-1 and U_n-1
    
      
    # solve linear system Ay=b AT_n = T_n-1 + CU_n-1, A is NOT constant matrix here
    A = A0 + lat*A_l
    
    A_x = lambda x: A@x
    mvec = la.LinearOperator((nv, nv), A_x)
   
    y,indix = la.gmres(mvec,b,tol=1e-7,M=pvec)        # T_n
    y=np.reshape(y,((nv,1)))
    # latent heat
    y_l,A_l=imex_latent(y)
    # up boundary condition
    Tu = y[0:-1:ny]
    U = Up_boundary(t,h,Tu)   # U_n
    #plug in right hand
    bU[0:-1:ny] = U
    
    t += dt
    if (i+1)%200==0:
       k = round((i+1)/200)
       print('=================================')
       print('now time is ',t)  
       Temp[:,[k]] = y
       
Tf = np.reshape(y,(ny,nx),order='F')
#plot
plt.imshow(Tf,cmap=plt.get_cmap('hsv'))

save('temp_lat_gmres.mat',{'Templ': Temp})






