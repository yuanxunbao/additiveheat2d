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
import sys
import os
from math import pi
import numpy as np
from ds_input_scn import phy_parameter, simu_parameter
from fir_deri_FD import gradscalar, gradflux
import time
from scipy.io import savemat as save
from numpy import shape

def add_BCs(u,BCx,BCy):
        
        m,n = u.shape
        ub = np.zeros((m+2,n+2))
        
        ub[1:-1,1:-1] = u
        
        if BCx == 'P':
                     
            ub[:,0] = ub[:,-2] 
            ub[:,-1] = ub[:,1]
            
            
        if BCy == 'P':
            
            ub[0,:] = ub[-2,:]
            ub[-1,:] = ub[1,:]
                
       
        if BCx == 'R':  # here just no flux, make the code more general
            
            ub[:,0] = ub[:,2]
            ub[:,-1] = ub[:,-3]        
            
        if BCy == 'R':
            
            ub[0,:] = ub[2,:]
            ub[-1,:] = ub[-3,:]
        
        
        #print(ub)
        return ub

# the equations are divergence of fluxs, having the form F_x + J_z 
def initial(): # properties/parameters of interests
    
    z0 = n.z0
    mag = n.mag
    dstb1 = mag*np.cos(2*pi/lx*xx*3)
    dstb2 = mag*np.cos(2*pi/lx*xx*5-10)
    dstb3 = -mag*np.cos(2*pi/lx*xx*7-20)
    psi0 = - sqrt2*( zz-z0+dstb1 +dstb2+dstb3) 
   
    U0 = 0*psi0 - 1
    
    
    return psi0,U0

def out_put():
    yo = np.zeros((2*nv,nts+1))
    for i in range(nts+1):
    
       phi = np.reshape(Tishot[:nv,i],(nz,nx))
       c_tilde = np.reshape(Tishot[nv:,i],(nz,nx))  

       yo[:,i] = np.reshape(np.vstack((np.reshape(phi,(nv,1),order='F'),np.reshape(c_tilde,(nv,1),order='F'))),(nv*2))

    filename = 'finallx'+ str(lx)+'.mat'

    save(filename,{'xx':xx,'zz':zz,'y':yo,'nx':nx,'nz':nz,'Mt':Mt})
    
    
    
    return


def mask_divi(A,B,eps):  # this routine should be able to solve a/b
    
    mask = 1*(B>eps)
    
    return mask*A/( B + (1-mask)*eps )



def normal(phi):
    
    phi = add_BCs(phi, 'P', 'R')
    
    phi_xx = s.gradxx(phi); phi_zx = s.gradzx(phi)
    phi_nx = np.sqrt( phi_xx**2 + phi_zx**2 )
    nxx = mask_divi(phi_xx, phi_nx, p.eps); nzx = mask_divi(phi_zx, phi_nx, p.eps)
    
    phi_xz = s.gradxz(phi); phi_zz = s.gradzz(phi)
    phi_nz = np.sqrt( phi_xz**2 + phi_zz**2 )
    nxz = mask_divi(phi_xz, phi_nz, p.eps); nzz = mask_divi(phi_zz, phi_nz, p.eps)
    
    phi_xc = s.gradxc(phi); phi_zc = s.gradzc(phi)
    phi_nc = np.sqrt( phi_xc**2 + phi_zc**2 )
    nxc = mask_divi(phi_xc, phi_nc, p.eps); nzc = mask_divi(phi_zc, phi_nc, p.eps)
    
    return nxx, nzx, nxz, nzz, nxc, nzc

def PF(nxx, nzx, nxz, nzz, psi_xx, psi_zx, psi_xz, psi_zz):

    nxx2 = nxx**2; nzx2 = nzx**2;
    nxz2 = nxz**2; nzz2 = nzz**2;
    a_sx = atheta(nxx2,nzx2)
    a_sz = atheta(nxz2,nzz2)
    aps_x = -16*p.delta*( nxx*nxx2*nzx - nxx*nzx*nzx2 )
    aps_z = -16*p.delta*( nxz*nxz2*nzz - nxz*nzz*nzz2 )
    
    
    F_x = a_sx**2*psi_xx - aps_x*a_sx*psi_zx

    J_z = a_sz**2*psi_zz + aps_z*a_sz*psi_xz


    return f.gradx( F_x ) + f.gradz( J_z ) 




def U_div(phi, U, jat, nxx, nzz):
    
    U =  add_BCs(U, 'P', 'R') 
    phi =  add_BCs(phi, 'P', 'R') 
    flux =  add_BCs(jat, 'P', 'R')
    
    
    diffx = p.Dl_tilde*(1- s.avgx(phi) )*s.gradxx(U)
    
    jatx = s.avgx(flux)*nxx   
    
    diffz = p.Dl_tilde*(1- s.avgz(phi) )*s.gradzz(U)
    
    jatz = s.avgz(flux)*nzz   
    
    
    return f.gradx( diffx + jatx ) + f.gradz( diffz + jatz )



def rhs_plapp(y,t):
    
    psi = np.reshape(y[:nv], (nz,nx))
    U = np.reshape(y[nv:], (nz,nx))
    phi = np.tanh(psi/sqrt2)
    
    psib = add_BCs(psi, 'P', 'R')
    
    nxx, nzx, nxz, nzz, nxc, nzc = normal(phi)
    
    
    psi_xx = s.gradxx(psib)
    psi_zx = s.gradzx(psib)
    
    psi_xz = s.gradxz(psib)
    psi_zz = s.gradzz(psib)

    psi_xc = s.gradxc(psib)
    psi_zc = s.gradzc(psib)
    psin2 = psi_xc**2 + psi_zc**2
            
    phi_div = PF(nxx, nzx, nxz, nzz, psi_xx, psi_zx, psi_xz, psi_zz)
    
    a_s2 = atheta(nxc**2,nzc**2)**2
    extra = -sqrt2*a_s2*phi*psin2
    
    Up = (zz - p.R_tilde*t)/p.lT_tilde
    rhs_psi = phi_div + extra + phi*sqrt2 - p.lamd*(1-phi**2)*sqrt2*( U + Up )
    tau_psi = ( 1- (1-p.k)*Up )*a_s2
       
    psi_t = rhs_psi/tau_psi

    jat = 0.5*( 1+ (1-p.k)*U )*( 1- phi**2 )*psi_t
    rhs_U = U_div(phi, U, jat, nxx, nzz) + sqrt2*jat

    tau_U = 1+p.k - (1-p.k)*phi 
    U_t = rhs_U/tau_U
   
    psi_t = np.reshape(psi_t,(nv,1))
    U_t = np.reshape(U_t,(nv,1))
    
    return np.vstack((psi_t,U_t))




#====================parameters/operators==========================
#y = load('W00.2lx6.0.mat')['y']
#t = load('plapp.mat')['t']
sqrt2 = np.sqrt(2.0)



p = phy_parameter()
n = simu_parameter()




lx_ex = n.lx
lx = lx_ex/p.W0

ratio = n.aratio
lz = ratio*lx


nx = n.nx
nz = int(ratio*nx+1)

nv= nz*nx #number of variables


dx = lx/nx
dz = lz/(nz-1)  

x = np.linspace(0,lx-dx,nx)

z = np.linspace(0,lz,nz)

xx, zz = np.meshgrid(x, z)
dt = n.dt


Mt = n.Mt
t=0



s = gradscalar(dx, dz)

f = gradflux(dx, dz)


atheta = lambda nx2, nz2: 1 - 3*p.delta + 4*p.delta*( nx2**2 + nz2**2 ) 


# =================Sampling parameters==================
nts = n.nts; #dts = Tscan/nts 
kts = int( Mt/nts )

##====================Initial condition=========================
Tishot = np.zeros((2*nv,nts+1))

psi0,C0 = initial()

y = np.vstack((np.reshape(psi0,(nv,1)), np.reshape(C0,(nv,1)) )) #T0/y0
Tishot[:,[0]] = y



#======================time evolusion=======================
start = time.time()

for i in range(Mt): #Mt
    
    rhs = rhs_plapp(y,t)
    
    y = y + dt*rhs #forward euler
    
  
    t += dt
    if (i+1)%kts==0:     # data saving 
       k = int(np.floor((i+1)/kts))
       print('=================================')
       print('now time is ',t)  
       Tishot[:,[k]] = y
       
end = time.time()
filename = n.filename

save(os.path.join(n.direc,filename),{'xx':xx*p.W0,'zz':zz*p.W0,'y':Tishot,'dt':dt*p.tau0,'nx':nx,'nz':nz,'t':t*p.tau0,'mach_time':end-start,'input_file':sys.argv[1]})




#out_put()




