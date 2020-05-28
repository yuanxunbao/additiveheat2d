#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:47:26 2020

@author: yigongqin
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:45:21 2020

@author: yigongqin
"""
# horizontal periodic/vertical no flux N*(M+1)
# Finite difference spatial second order
# Time rk45
# grids x*y N*(M+1) matix (m+1)*n

from math import pi
import numpy as np
from SCN import parameters 
from fir_deri_FD import gradscalar, gradflux

import matplotlib.pyplot as plt
from scipy.io import savemat as save
from scipy.io import loadmat as load

# the equations are divergence of fluxs, having the form F_x + J_z 
def initial(): # properties/parameters of interests
  
    z0 = lx*0.1
    nw = 1
    mag = 0.2

   
    
    phi0 = -np.tanh( ( zz-z0-mag*np.cos(2*pi/lx*xx*nw) ))
   

    #U0 = 1/(1-p.k) * ( p.k/((1-phi0)/2+p.k*(1+phi0)/2) - 1)
    U0 = 0*phi0 - 1
    
    return phi0,U0

def out_put():
    yo = np.zeros((2*nv,nts+1))
    for i in range(nts+1):
    
       phi = np.reshape(Tishot[:nv,i],(nz,nx))
       U = np.reshape(Tishot[nv:,i],(nz,nx))  
       c_tilde = ( 1+ (1-p.k)*U )*( (1+phi)/2 + (1-phi)/(2*p.k) )
       yo[:,i] = np.reshape(np.vstack((np.reshape(phi,(nv,1),order='F'),np.reshape(c_tilde,(nv,1),order='F'))),(nv*2))

    filename = 'a' + str(int(alpha0*180/pi))+'W0' + str(p.W0)+'lx'+ str(lx*p.W0)+'.mat'

    save(filename,{'xx':xx*p.W0,'zz':zz*p.W0,'y':yo,'dt':dt*p.tau0,'nx':nx,'nz':nz,'t':t*p.tau0})
    
    
    
    return

def misorien(phi_x, phi_z):
    

    phi_xr = cosa*phi_x + sina*phi_z
    phi_zr = -sina*phi_x + cosa*phi_z
    
    return phi_xr,phi_zr


def mask_divi(A,B,eps):  # this routine should be able to solve a/b
    
    mask = 1*(B>eps)
    
    return mask*A/( B + (1-mask)*eps )

def PF(phi_xx,phi_zx,phi_nx,phi_xz,phi_zz,phi_nz):

    phi_xxr,phi_zxr = misorien(phi_xx,phi_zx)
    phi_xzr,phi_zzr = misorien(phi_xz,phi_zz)
   # print(phi_xx)
    
    a_sx = 1 - 3*p.delta + mask_divi( 4*p.delta*( phi_xxr**4 + phi_zxr**4 ) , phi_nx**4 , p.eps**4)
    a_sz = 1 - 3*p.delta + mask_divi( 4*p.delta*( phi_xzr**4 + phi_zzr**4 ) , phi_nz**4 , p.eps**4)
    aps_x = mask_divi( -16*p.delta*( phi_xxr**3*phi_zxr - phi_xxr*phi_zxr**3 ) , phi_nx**4 , p.eps**4)
    aps_z = mask_divi( -16*p.delta*( phi_xzr**3*phi_zzr - phi_xzr*phi_zzr**3 ) , phi_nz**4 , p.eps**4)
    
    
    F_x = a_sx**2*phi_xx - aps_x*a_sx*phi_zx

    J_z = a_sz**2*phi_zz + aps_z*a_sz*phi_xz


    return f.gradx( F_x ) + f.gradz( J_z )




def FU(phi_x, phi_n, phi, U, phi_t):
    
    diffx = p.Dl_tilde/2*( 1 - s.avgx(phi) )*s.gradxx(U) 
    jatx = mask_divi( p.alpha* s.avgx( ( 1+ (1-p.k)*U )*phi_t )*phi_x , phi_n , p.eps)
    
    return diffx + jatx

def JU(phi_z, phi_n, phi, U, phi_t):
    
    diffz = p.Dl_tilde/2*( 1 - s.avgz(phi) )*s.gradzz(U)
    jatz = mask_divi( p.alpha* s.avgz( ( 1+ (1-p.k)*U )*phi_t )*phi_z , phi_n , p.eps)
    
    return diffz + jatz


def a_s(phi_x, phi_z):
    
    phi_xc = f.avx(phi_x)
    phi_zc = f.avz(phi_z)
    phi_n = np.sqrt( phi_xc**2 + phi_zc**2 )
    #print(phi_n)
    phi_xc,phi_zc = misorien(phi_xc,phi_zc)
    #phi_n1 = np.sqrt( phi_xc**2 + phi_zc**2 )
    #print(phi_n1)
    return 1 - 3*p.delta + mask_divi( 4*p.delta*( phi_xc**4 + phi_zc**4 ) , phi_n**4 , p.eps**4)


def rhs_plapp(y,t):
    
    phi = np.reshape(y[:nv], (nz,nx))
    U = np.reshape(y[nv:], (nz,nx))

    phi_xx = s.gradxx(phi)
    phi_zx = s.gradzx(phi)
    
    phi_nx = np.sqrt( phi_xx**2 + phi_zx**2 )
    phi_xz = s.gradxz(phi)
    phi_zz = s.gradzz(phi)
    
    phi_nz = np.sqrt( phi_xz**2 + phi_zz**2 )
    

    
    phi_div = PF(phi_xx,phi_zx,phi_nx,phi_xz,phi_zz,phi_nz)
    
    rhs_phi = phi_div + phi - phi**3 - p.lamd*(1-phi**2)**2*( U+ (zz-p.R_tilde*t)/p.lT_tilde )
    tau_phi = a_s( phi_xx, phi_zz )**2*( 1- (1-p.k)*(zz-p.R_tilde*t)/p.lT_tilde )
    

    
    phi_t = rhs_phi/tau_phi
    
    U_div = f.gradx( FU(phi_xx,phi_nx,phi,U,phi_t) ) + f.gradz( JU(phi_zz,phi_nz,phi,U,phi_t) )
    rhs_U = U_div + ( 1+ (1-p.k)*U )/2*phi_t
    tau_U = (1+p.k)/2 - (1-p.k)/2*phi
    U_t = rhs_U/tau_U
    
    phi_t = np.reshape(phi_t,(nv,1))
    U_t = np.reshape(U_t,(nv,1))
    
    return np.vstack((phi_t,U_t))




#====================parameters/operators==========================
#y = load('W00.2lx6.0.mat')['y']
#t = load('plapp.mat')['t']

p = parameters()


alpha0 = -pi/180*30
cosa = np.cos(alpha0)
sina = np.sin(alpha0)


lx_ex = 22.5
lx = lx_ex/p.W0

ratio = 32
lz = ratio*lx


nx = 20
nz = ratio*nx+1

nv= nz*nx #number of variables


dx = lx/nx
dz = lz/(nz-1)  

x = np.linspace(0,lx-dx,nx)

z = np.linspace(0,lz,nz)

xx, zz = np.meshgrid(x, z)
dt = 0.0005



Mt = 120000
t=0



s = gradscalar(dx, dz, 'P', 'R')

f = gradflux(dx, dz)



# =================Sampling parameters==================
nts = 50; #dts = Tscan/nts 
kts = int( Mt/nts )

##====================Initial condition=========================
Tishot = np.zeros((2*nv,nts+1))

phi0,U0 = initial()

y = np.vstack((np.reshape(phi0,(nv,1)), np.reshape(U0,(nv,1)) )) #T0/y0
Tishot[:,[0]] = y


fig0 =plt.figure()
ax01 = fig0.add_subplot(121)
plt.title('phi')
plt.imshow(phi0,cmap=plt.get_cmap('winter'),origin='lower')
ax02 = fig0.add_subplot(122)
plt.title('U')
plt.imshow(U0,cmap=plt.get_cmap('winter'),origin='lower')
#======================time evolusion=======================

for i in range(Mt): #Mt
    
    rhs= rhs_plapp(y,t)
    
    y = y + dt*rhs #forward euler
    
  
    t += dt
    if (i+1)%kts==0:     # data saving 
       k = int(np.floor((i+1)/kts))
       print('=================================')
       print('now time is ',t)  
       Tishot[:,[k]] = y
       
phif = np.reshape(y[:nv],(nz,nx))
Uf = np.reshape(y[nv:],(nz,nx))
fig1 = plt.figure()
# 
ax11 = fig1.add_subplot(121)
plt.title('phi')
plt.imshow(phif,cmap=plt.get_cmap('winter'),origin='lower')
ax12 = fig1.add_subplot(122)
plt.title('U')
plt.imshow(Uf,cmap=plt.get_cmap('winter'),origin='lower')
# #Tfc = np.reshape(y[ins],(nys,nxs),order='F')

#Tfc = np.reshape(y[ins],(nys,nxs),order='F')
#plot
#plt.imshow(Tfc,cmap=plt.get_cmap('hsv'))


out_put()




