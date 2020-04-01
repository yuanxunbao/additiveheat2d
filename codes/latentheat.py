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
import time
#from scipy.sparse import csc_matrix
class gmres_counter(object):
    def __init__(self, TOL):
       # self._disp = True
        self.niter = 0
        self.tol = TOL
    def __call__(self, rk=None):
        self.niter += 1
        if rk<self.tol:
            print('gmres iteration %3i\trk = %s' % (self.niter, str(rk)))
            
def heat_source(t,h):
    #print('nx=',nx,'amazing!')
    
    return p.q0 * np.exp( -2*( np.arange(nx).reshape((nx,1))*h - l0 - p.Vs*t )**2/p.rb**2 )
    

def Up_boundary(t,h,Tu):
    
    qs = heat_source(t,h)
    
    return -2*h/p.K*( -qs + p.hc*(Tu-p.Te) + p.epsilon*p.sigma*(Tu**4-p.Te**4) )

def imex_latent(y):
    
    down = y>p.Ts
    up = y<p.Tl
    
    y_l = y*down*up
    ind = np.ones((nv,1))*down*up
    ind = np.reshape(ind,(nv))
    A_l = sp.diags(ind,format='csc')
    
    return y_l,A_l


start = time.time()

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

lat = p.Lm/(p.Tl-p.Ts)  # latent heat

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

A0 = I - C*L
#Aarr = A.toarray()

#pd =is_pos_def(A.toarray())   #check if A matrix is positive definite 
# LU decomposition Pr*A*Pc = LU
lu = la.splu(A0)

LU_x = lambda x: lu.solve(x)
pvec = la.LinearOperator((nv, nv), LU_x)


##====================Initial condition=========================
Temp = np.zeros((nvs,nts+1))

#T = np.zeros((ny,nx))      # temperature T0
#T = np.arange(nv).reshape((ny,nx))
T0 = 1*np.ones((ny,nx))
y = np.reshape(T0,(nv,1),order='F')  #T0/y0
Temp[:,[0]] = y[ins]

y_l,A_l=imex_latent(y)
#plt.imshow(T0,cmap=plt.get_cmap('hot'))

Tu = y[0:-1:ny]
U = Up_boundary(t,h,Tu)          # U0

bU = np.zeros((nv,1))
bU[0:-1:ny] = U

#======================time evolusion=======================

TOL = 1e-9 #gmres tolerance


for i in range(Mt):
    
    # obtain right hand side
    b = y + lat*y_l + C*bU           # b_n-1 from T_n-1 and U_n-1
    
    
    A = A0 + lat*A_l
    A_x = lambda x: A@x
    mvec = la.LinearOperator((nv, nv), A_x)
    
    counter = gmres_counter(TOL)
    y,indix = la.gmres(A,b,x0=y,tol=TOL,M=pvec,callback=counter)        # T_n
    y=np.reshape(y,((nv,1)))
    
    # latent heat
    y_l,A_l=imex_latent(y)    
   # Up boundary
    Tu = y[0:-1:ny]
    U = Up_boundary(t,h,Tu)   # U_n
    #plug in right hand
    bU[0:-1:ny] = U
    
    t += dt
    if (i+1)%kts==0:
       k = int(np.floor((i+1)/kts))
       print('=================================') 
       print('now time is ',t) 
       print('=================================') 
       Temp[:,[k]] = y[ins]
       #print('gmres iterations: ',counter)
Tf = np.reshape(y,(ny,nx),order='F')
#plot
plt.imshow(Tf,cmap=plt.get_cmap('hsv'))



filename = 'latdt' + str(dt)+'xgrids'+str(nx)+'.mat'
tempname = 'temp'+str(nx)
save(filename,{tempname: Temp})


plt.spy(lu.L)

end =time.time()
print('time used: ',end-start)

