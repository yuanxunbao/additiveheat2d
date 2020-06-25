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
# Time FE
# grids x*y N*(M+1) matix (m+1)*n
import os
import time

os.environ["NUMBA_NUM_THREADS"] = '2'

from math import pi
import numpy as np
from numpy.random import rand
from dsinput_Takaki import phys_para, simu_para, IO_para
from grad_routine_jit import gradxx,gradzx,gradxz,gradzz,gradxc,gradzc,avgx,avgz,fgradx,fgradz,norm2d
from numba import njit
from scipy.io import savemat as save

pflag = True

@njit(parallel=pflag)
def atheta(nx2, nz2):
    
    return 1 - 3*delta + 4*delta*( nx2**2 + nz2**2 )  
    
@njit(parallel=pflag)
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
                
        return ub

def initial(): # properties/parameters of interests
    
    r0 = 0.5625
    r = np.sqrt( (xx-lx/2) **2+zz**2) - r0    
    psi0 = - r 
    U0 = 0*psi0 -0.3
    
    return psi0,U0

def reshape_data(y):
    yo = np.zeros((2*nv,1))

    phi = np.tanh( np.reshape(y[:nv,0],(nz,nx))/ sqrt2)
    U = np.reshape(y[nv:,0],(nz,nx))  
    c_tilde = ( 1+ (1-k)*U )*( (1+phi)/2 + (1-phi)/(2*k) )
    yo = np.vstack((np.reshape(phi,(nv,1),order='F'),np.reshape(c_tilde,(nv,1),order='F')))
     
    return yo

@njit(parallel=pflag)
def misorien(phi_x, phi_z):
    

    phi_xr = cosa*phi_x + sina*phi_z
    phi_zr = -sina*phi_x + cosa*phi_z
    
    return phi_xr,phi_zr

@njit(parallel=pflag)
def noise(y):

    beta  = rand(nv,1) - 0.5
    
    noi = eta*np.sqrt(dt)*beta

    return np.vstack((noi, np.zeros((nv,1))))

@njit(parallel=pflag)
def mask_divi(A,B,eps):  # this routine should be able to solve a/b
    
    mask = 1*(B>eps)
    
    return mask*A/( B + (1-mask)*eps )

@njit(parallel=pflag)
def tau_psi_mask(tau_psi):
    mask = 1*(tau_psi>k)
    return mask*tau_psi + (1-mask)*k

@njit()
def normal(phi):
    
    phi = add_BCs(phi, 'P', 'R')
    
    phi_xx = gradxx(hi,phi); phi_zx = gradzx(hi,phi)
    phi_nx = norm2d( phi_xx, phi_zx )
    nxx = mask_divi(phi_xx, phi_nx, eps); nzx = mask_divi(phi_zx, phi_nx, eps)
    nxxr,nzxr = misorien(nxx,nzx)
    
    phi_xz = gradxz(hi,phi); phi_zz = gradzz(hi,phi)
    phi_nz = norm2d( phi_xz, phi_zz )
    nxz = mask_divi(phi_xz, phi_nz, eps); nzz = mask_divi(phi_zz, phi_nz, eps)
    nxzr,nzzr = misorien(nxz,nzz)
    
    phi_xc = gradxc(hi,phi); phi_zc = gradzc(hi,phi)
    phi_nc = norm2d( phi_xc, phi_zc )
    nxc = mask_divi(phi_xc, phi_nc, eps); nzc = mask_divi(phi_zc, phi_nc, eps)
    nxcr,nzcr = misorien(nxc,nzc)
    
    return nxxr, nzxr, nxzr, nzzr, nxcr, nzcr, nxx, nzz


@njit(parallel=pflag)
def Phase_div(nxx, nzx, nxz, nzz, psi_xx, psi_zx, psi_xz, psi_zz):

    nxx2 = nxx**2; nzx2 = nzx**2;
    nxz2 = nxz**2; nzz2 = nzz**2;
    a_sx = atheta(nxx2,nzx2)
    a_sz = atheta(nxz2,nzz2)
    aps_x = -16*delta*( nxx*nxx2*nzx - nxx*nzx*nzx2 )
    aps_z = -16*delta*( nxz*nxz2*nzz - nxz*nzz*nzz2 )
    
    
    F_x = a_sx**2*psi_xx - aps_x*a_sx*psi_zx

    J_z = a_sz**2*psi_zz + aps_z*a_sz*psi_xz


    return fgradx( hi, F_x ) + fgradz( hi, J_z ) 

@njit(parallel=pflag)
def U_div2(phi,U):

    U =  add_BCs(U, 'P', 'R')
    phi = add_BCs(phi, 'P', 'R')
    diffx = Dl_tilde*(1- avgx(phi) )*gradxx(hi, U)

    return diffx

@njit(parallel=pflag)
def U_div(phi, U, jat, nxx, nzz):
    
    U =  add_BCs(U, 'P', 'R') 
    phi =  add_BCs(phi, 'P', 'R') 
    flux =  add_BCs(jat, 'P', 'R')
    
    
    diffx = Dl_tilde*(1- avgx(phi) )*gradxx(hi, U)
    
    jatx = avgx(flux)*nxx   
    
    diffz = Dl_tilde*(1- avgz(phi) )*gradzz(hi, U)
    
    jatz = avgz(flux)*nzz   
    

    return fgradx( hi, diffx + jatx ) + fgradz( hi, diffz + jatz )


def rhs_dirsolid(y,t):
    
    psi = np.reshape(y[:nv], (nz,nx))
    U = np.reshape(y[nv:], (nz,nx))
    
    phi = np.tanh(psi/sqrt2)

    psib = add_BCs(psi, 'P', 'R')
    
    nxx, nzx, nxz, nzz, nxc, nzc, nxxo, nzzo = normal(phi)
    
    
    psi_xx = gradxx(hi,psib)
    psi_zx = gradzx(hi,psib)
    
    psi_xz = gradxz(hi,psib)
    psi_zz = gradzz(hi,psib)

    psi_xc = gradxc(hi,psib)
    psi_zc = gradzc(hi,psib)
    psin2 = psi_xc**2 + psi_zc**2
            
    psi_div = Phase_div(nxx, nzx, nxz, nzz, psi_xx, psi_zx, psi_xz, psi_zz)
    
    a_s2 = atheta(nxc**2,nzc**2)**2
    extra = -sqrt2*a_s2*phi*psin2
    
    Up = (zz - R_tilde*t)/lT_tilde
    rhs_psi = psi_div + extra + phi*sqrt2 - lamd*(1-phi**2)*sqrt2*( U + Up )
    tau_psi = tau_psi_mask( ( 1- (1-k)*Up ) )*a_s2
       
    psi_t = rhs_psi/tau_psi

    jat = 0.5*( 1+ (1-k)*U )*( 1- phi**2 )*psi_t
    rhs_U = U_div(phi, U, jat, nxxo, nzzo) + sqrt2*jat

    tau_U = 1+k - (1-k)*phi 
    U_t = rhs_U/tau_U
   
    psi_t = np.reshape(psi_t,(nv,1))
    U_t = np.reshape(U_t,(nv,1))
    
    return np.vstack((psi_t,U_t))




#====================parameters/operators==========================

sqrt2 = np.sqrt(2.0)


delta, k, lamd, R_tilde, Dl_tilde, lT_tilde, W0, tau0 = phys_para()
eps, alpha0, lxd, aratio, nx, dt, Mt, eta, filename = simu_para(W0,Dl_tilde)
nts, direc, r0 = IO_para(W0,lxd)

alpha0 = alpha0*pi/180

cosa = np.cos(alpha0)
sina = np.sin(alpha0)

lx = lxd/W0

ratio = aratio
lz = ratio*lx

nz = int(ratio*nx+1)

nv= nz*nx #number of variables


dx = lx/nx
dz = lz/(nz-1)  

x = np.linspace(0,lx-dx,nx)

z = np.linspace(0,lz,nz)

xx, zz = np.meshgrid(x, z)

hi = 1.0/dx

t=0



# =================Sampling parameters==================
kts = int( Mt/nts )

##====================Initial condition=========================
Tishot = np.zeros((2*nv,nts+1))
np.random.seed(1)

psi0,U0 = initial()

y = np.vstack((np.reshape(psi0,(nv,1)), np.reshape(U0,(nv,1)) )) #

#rhs_dirsolid(y,t)
# normal(psi0)
nxx, nzx, nxz, nzz, nxc, nzc, nxxo, nzzo = normal(psi0)
U_div(psi0,U0,U0,nxx,nzz)
U_div2(psi0,U0)

U =  add_BCs(U0, 'P', 'R')
phi =  add_BCs(psi0, 'P', 'R')
flux =  add_BCs(U0, 'P', 'R')


diffx = Dl_tilde*(1- avgx(phi) )*gradxx(hi, U)
jatx = avgx(flux)*nxx
diffz = Dl_tilde*(1- avgz(phi) )*gradzz(hi, U)
jatz = avgz(flux)*nzz
dd = fgradx( hi, diffx + jatx ) + fgradz( hi, diffz + jatz )


Tishot[:,[0]] = reshape_data(y)


#======================time evolusion=======================

start = time.time()

for ii in range(Mt): #Mt

#    zzz = gradxx(hi,U0)
    #rhs = normal(psi0)
    U =  add_BCs(U0, 'P', 'R')
    phi =  add_BCs(psi0, 'P', 'R')
#    flux =  add_BCs(U0, 'P', 'R')


    diffx = Dl_tilde*(1- avgx(phi) )*gradxx(hi, U)

#    jatx = avgx(flux)*nxx

#    diffz = Dl_tilde*(1- avgz(phi) )*gradzz(hi, U)

#    jatz = avgz(flux)*nzz

#    dd = fgradx( hi, diffx + jatx ) + fgradz( hi, diffz + jatz )
#    y = y + dt*rhs + noise(y)#forward euler
 	  
#    t += dt
#    if (ii+1)%kts==0:     # data saving 
#       kk = int(np.floor((ii+1)/kts))
#       print('=================================')
#       print('now time is ',t)  
#       Tishot[:,[kk]] = reshape_data(y)
       
       
end = time.time() 
print('time = ', end -start)



start2 = time.time()

for ii in range(Mt):
    
    U_div2(psi0,U0)

end2 = time.time()


print('time = ', end2 -start2)

# save(os.path.join(direc,filename),{'xx':xx*W0,'zz':zz*W0,'y':Tishot,'dt':dt*tau0,'nx':nx,'nz':nz,'t':t*tau0,'mach_time':end-start})




