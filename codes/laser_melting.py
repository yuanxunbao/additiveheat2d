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

def heat_source(t,h):
    #print('nx=',nx,'amazing!')
    
    return p.q0 * np.exp( -2*( np.arange(nx).reshape((nx,1))*h - l0 - p.Vs*t )**2/p.rb**2 )
    

def Up_boundary(t,h,Tu):
    
    qs = heat_source(t,h)
    
    return -2*h/p.K*( -qs + p.hc*(Tu-p.Te) + p.epsilon*p.sigma*(Tu**4-p.Te**4) )



#====================parameters==========================

p = parameters()

lx = 8
aratio = 4
ly = lx/aratio

print ('enter x grids nx: ') 
nx =int(input()) 

h = lx/(nx-1)
print('mesh width is', h)

ny = int((nx-1)/aratio+1)
print('y grids ny:',ny)
#N = 500;M = 100
#nx= N+1  #x grids;ny= M+1  #y grids
nv= ny*nx #number of variables

print ('enter time step dt: ') 
dt =float(input()) 
C = p.alpha*dt/h**2  #CFL number

l0 = lx/8 #laser starting point 
lscan = lx - 2*l0
Tscan = lscan/p.Vs   #scanning stop time
Mt= int(Tscan/dt)  #time steps
t=0

# Sampling parameters
nts = 50; dts = Tscan/nts 
kts = int( Mt/nts )
print('enter sample x grids: ')
nxs = int(input()) 
dns = int( (nx-1)/(nxs-1) )
nys = int((nxs-1)/aratio+1); nvs=nys*nxs
ins = np.zeros((nys,nxs),dtype=int)
ins[[0],:] = np.arange(0,nv-1,ny*dns)
for i in range(1,nys):
    ins[[i],:]=ins[[i-1],:]+dns
ins = np.reshape(ins,(nvs),order='F') 

save('para.mat',{'lx': lx,'ly': ly,'nxs': nxs,'nys': nys,'dts': dts,'nts': nts})
#=====================Generate matrix========================

I = sp.eye(nx*ny,format='csc')

Dxx = sp.diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx)).toarray()
Dxx[0,1]=2; Dxx[-1,-2]=2 #Riemann boundary condition 

Dyy = sp.diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny)).toarray()
Dyy[0,1]=2; Dyy[-1,-2]=2


L = sp.kronsum(Dyy,Dxx,format='csc')

A = I - C*L

# LU decomposition Pr*A*Pc = LU
lu = la.splu(A)


##====================Initial condition=========================
Temp = np.zeros((nvs,nts+1))

#T = np.zeros((ny,nx))      # temperature T0
#T = np.arange(nv).reshape((ny,nx))
T0 = 1*np.ones((ny,nx))
y = np.reshape(T0,(nv,1),order='F')  #T0/y0
Temp[:,[0]] = y[ins]

#plt.imshow(T0,cmap=plt.get_cmap('hot'))

Tu = y[0:-1:ny]
U = Up_boundary(t,h,Tu)          # U0

bU = np.zeros((nv,1))
bU[0:-1:ny] = U

#======================time evolusion=======================

for i in range(Mt):
    
    # obtain right hand side
    b = y + C*bU            # b_n-1 from T_n-1 and U_n-1
    
    # solve linear system Ay=b AT_n = T_n-1 + CU_n-1, A is constant matrix here
    y = lu.solve(b)        # T_n
    
   # Sample from T
    Tu = y[0:-1:ny]
    U = Up_boundary(t,h,Tu)   # U_n
    #plug in right hand
    bU[0:-1:ny] = U
    
    t += dt
    if (i+1)%kts==0:
       k = int(np.floor((i+1)/kts))
       print('=================================')
       print('now time is ',t)  
       Temp[:,[k]] = y[ins]
       
Tf = np.reshape(y,(ny,nx),order='F')
#plot
plt.imshow(Tf,cmap=plt.get_cmap('hsv'))

#Tfc = np.reshape(y[ins],(nys,nxs),order='F')
#plot
#plt.imshow(Tfc,cmap=plt.get_cmap('hsv'))

filename = 'nolatdt' + str(dt)+'xgrids'+str(nx)+'.mat'
tempname = 'temp'+str(nx)
#save(filename,{tempname: Temp})






