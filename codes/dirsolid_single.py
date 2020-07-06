#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:10:13 2020
Single precision: there should be no int/float64 in calculation

@author: yigong qin, yuanxun bao
"""
import importlib
import sys
import os
from scipy.io import savemat as save
from numba import njit, stencil, vectorize, float32, float64
import numpy as np
from numpy.random import random,rand
import time
from math import pi
#PARA = importlib.import_module(sys.argv[1])
import dsinput_single as PARA
from numpy import float32 as f32

pi = f32(pi)
s0 = f32(0); s1 = f32(1); s2 =f32(2); p5 = f32(0.5); p25 = f32(0.25)
delta, k, lamd, R_tilde, Dl_tilde, lT_tilde, W0, tau0 = PARA.phys_para()
eps, alpha0, lxd, aratio, nx, dt, Mt, eta, filename = PARA.simu_para(W0,Dl_tilde)
U_0, seed, nts, direc = PARA.IO_para(W0,lxd)

alpha0 = alpha0*pi/180; alpha0 = f32(alpha0)

cosa = np.cos(alpha0)
sina = np.sin(alpha0)

a_s = 1 - 3*delta  ; a_s = f32(a_s)
epsilon = 4*delta/a_s; epsilon = f32(epsilon)
a_12 = 4*a_s*epsilon;  a_12 = f32(a_12)
sqrt2 = np.sqrt(s2)

lx = lxd/W0

ratio = aratio
lz = ratio*lx

nz = int(ratio*nx+1)

nv= nz*nx #number of variables
dx = lx/nx
dz = lz/(nz-1)


x = np.linspace(0,lx-dx,nx,dtype=f32)

z = np.linspace(0,lz,nz,dtype=f32)

xx, zz = np.meshgrid(x, z)
t=s0


hi= 1./dx;   hi = f32(hi)
dt_sr = np.sqrt(dt)

Tishot = np.zeros((2*nv,nts+1),dtype=f32)

np.random.seed(seed)
eps2 = np.sqrt(eps)

# update global array, don't jit compile this function
def set_BC(u,BCx,BCy):
    
    # 0 periodic, 1 no flux (Neumann)
    
    if BCx == 0 :
                
        u[ 0,:] = u[-2,:] # 1st =  last - 1
        u[-1,:] = u[ 1,:] # last = second
        
    if BCx == 1 :
        
        u[ 0,:] = u[ 2,:] # 1st = 3rd
        u[-1,:] = u[-3,:] # last = last - 2
    
    if BCy == 0 :
        
        u[:, 0]  = u[:,-2]
        u[:,-1]  = u[:, 1]
        
    if BCy == 1 :
        
        u[:,0] = u[:,2]
        u[:,-1] = u[:,-3]
        
        
    return u

@njit
def set_halo(u):
    
    m,n = u.shape
    ub = np.zeros((m+2,n+2),dtype=f32)
    
    ub[1:-1,1:-1] = u
    
    return ub



@vectorize([float32(float32, float32)])
def atheta(ux, uz):

    ux2 = ( cosa*ux + sina*uz )**s2
    uz2 = ( -sina*ux + cosa*uz)**s2
        
    # return MAG_sq2
    MAG_sq2 = (ux2 + uz2)**s2
    
    if (MAG_sq2 > eps**s2):
        
        return a_s*( s1 + epsilon*(ux2**s2 + uz2**s2) / MAG_sq2   )
        
    else:
        return s1
    
    
@vectorize([float32(float32, float32)])
def aptheta(ux, uz):
    uxr = cosa*ux + sina*uz
    uzr = -sina*ux + cosa*uz
    ux2 = uxr**s2
    uz2 = uzr**s2
    
    MAG_sq2 = (ux2 + uz2)**s2
    
    if (MAG_sq2 > eps**s2):
        
        return -a_12*uxr*uzr*(ux2 - uz2) /  MAG_sq2
    
    else:
        return s0




@vectorize([float32(float32, float32, float32, float32)])
def divi(a, b, ep, bound):    
    nx = a / b if (b > ep) else bound    
    return nx


        

@stencil
def _rhs_psi(ps,ph,U,zz):
    #'''
    # ps = psi, ph = phi
    
    # =============================================================
    # 
    # 1. ANISOTROPIC DIFFUSION
    # 
    # =============================================================
    
    # these ps's are defined on cell centers
    psipjp=( ps[ 1, 1] + ps[ 0, 1] + ps[ 0, 0] + ps[ 1, 0] ) * p25
    psipjm=( ps[ 1, 0] + ps[ 0, 0] + ps[ 0,-1] + ps[ 1,-1] ) * p25
    psimjp=( ps[ 0, 1] + ps[-1, 1] + ps[-1, 0] + ps[ 0, 0] ) * p25
    psimjm=( ps[ 0, 0] + ps[-1, 0] + ps[-1,-1] + ps[ 0,-1] ) * p25
    
    phipjp=( ph[ 1, 1] + ph[ 0, 1] + ph[ 0, 0] + ph[ 1, 0] ) * p25
    phipjm=( ph[ 1, 0] + ph[ 0, 0] + ph[ 0,-1] + ph[ 1,-1] ) * p25
    phimjp=( ph[ 0, 1] + ph[-1, 1] + ph[-1, 0] + ph[ 0, 0] ) * p25
    phimjm=( ph[ 0, 0] + ph[-1, 0] + ph[-1,-1] + ph[ 0,-1] ) * p25

    # ============================
    # right edge flux
    # ============================
    psx = ps[1,0]-ps[0,0]
    psz = psipjp - psipjm
    phx = ph[1,0]-ph[0,0]
    phz = phipjp - phipjm
    
    A  = atheta( phx,phz)
    Ap = aptheta(phx,phz)
    JR = A * ( A*psx - Ap*psz )
    
    # ============================
    # left edge flux
    # ============================
    psx = ps[0,0]-ps[-1,0]
    psz = psimjp - psimjm
    phx = ph[0,0]-ph[-1,0]
    phz = phimjp - phimjm
    
    A  = atheta( phx,phz)
    Ap = aptheta(phx,phz)
    JL = A * ( A*psx - Ap*psz )
    
    # ============================
    # top edge flux
    # ============================
    psx = psipjp - psimjp
    psz = ps[0,1]-ps[0,0]
    phx = phipjp - phimjp
    phz = ph[0,1]-ph[0,0]


    A  = atheta( phx,phz)
    Ap = aptheta(phx,phz)
    JT = A * ( A*psz + Ap*psx )
    
    # ============================
    # bottom edge flux
    # ============================
    psx = psipjm - psimjm
    psz = ps[0,0]-ps[0,-1]
    phx = phipjm - phimjm
    phz = ph[0,0]-ph[0,-1]
    
    A  = atheta( phx,phz)
    Ap = aptheta(phx,phz)
    JB = A * ( A*psz + Ap*psx )
    

    # =============================================================
    # 
    # 2. EXTRA TERM: sqrt2 * atheta**2 * phi * |grad psi|^2
    # 
    # =============================================================
    
    # d(phi)/dx  d(psi)/dx d(phi)/dz  d(psi)/dz at nodes (i,j)
    phxn = ( ph[ 1, 0] - ph[-1, 0] ) * p5
    phzn = ( ph[ 0, 1] - ph[ 0,-1] ) * p5
    psxn = ( ps[ 1, 0] - ps[-1, 0] ) * p5
    pszn = ( ps[ 0, 1] - ps[ 0,-1] ) * p5
    
    A2 = atheta(phxn,phzn)**s2
    gradps2 = (psxn)**s2 + (pszn)**s2
    extra =  -sqrt2 * A2 * ph[0,0] * gradps2
    
    
    # =============================================================
    # 
    # 3. double well (transformed): sqrt2 * phi + nonlinear terms
    # 
    # =============================================================
    
    Up = (zz[0,0] )/lT_tilde
    
    rhs_psi = ((JR-JL) + (JT-JB) + extra) * hi**s2 + \
             sqrt2*ph[0,0] - lamd*(s1-ph[0,0]**s2)*sqrt2*(U[0,0] + Up) 
        
    
    # =============================================================
    # 
    # 4. dpsi/dt term
    # 
    # =============================================================
    tp = (s1-(s1-k)*Up)
    tau_psi = tp*A2 if tp >= k else k*A2
    

    return rhs_psi/tau_psi + eta*(f32(random())- p5 )/dt_sr



@stencil
def _rhs_U(U,ph,jat):
    
    # define cell centered values
    phipjp=( ph[ 1, 1] + ph[ 0, 1] + ph[ 0, 0] + ph[ 1, 0] ) * p25
    phipjm=( ph[ 1, 0] + ph[ 0, 0] + ph[ 0,-1] + ph[ 1,-1] ) * p25
    phimjp=( ph[ 0, 1] + ph[-1, 1] + ph[-1, 0] + ph[ 0, 0] ) * p25
    phimjm=( ph[ 0, 0] + ph[-1, 0] + ph[-1,-1] + ph[ 0,-1] ) * p25

    # ============================
    # right edge flux (i+1/2, j)
    # ============================
    phx = ph[1,0]-ph[0,0]
    phz = phipjp - phipjm
    phn = np.sqrt(phx**s2 + phz**s2) 
    nx = divi(phx, phn, eps2, s0)
    
    
    #jat    = p5*(s1+(s1-k)*U[0,0])*(s1-ph[0,0]**s2)*psi_t[0,0]
    #jat_ip = p5*(s1+(s1-k)*U[1,0])*(s1-ph[1,0]**s2)*psi_t[1,0]
        
    UR = hi*Dl_tilde*p5*(s2 - ph[0,0] - ph[1,0])*(U[1,0]-U[0,0]) + \
         p5*(jat[0,0] + jat[1,0])*nx
         
       
    # ============================
    # left edge flux (i-1/2, j)
    # ============================
    phx = ph[0,0]-ph[-1,0]
    phz = phimjp - phimjm
    phn = np.sqrt(phx**s2 + phz**s2) 
    nx = divi(phx, phn, eps2, s0)
    
    #jat_im = p5*(s1+(s1-k)*U[-1,0])*(s1-ph[-1,0]**s2)*psi_t[-1,0]

    UL = hi*Dl_tilde*p5*(s2 - ph[0,0] - ph[-1,0])*(U[0,0]-U[-1,0]) + \
         p5*(jat[0,0] + jat[-1,0])*nx
         
     
    # ============================
    # top edge flux (i, j+1/2)
    # ============================     
    phx = phipjp - phimjp
    phz = ph[0,1]-ph[0,0]
    phn = np.sqrt(phx**s2 + phz**s2) 
    nz = divi(phz, phn, eps2, s0)
          
    #jat_jp = p5*(s1+(s1-k)*U[0,1])*(s1-ph[0,1]**s2)*psi_t[0,1]      
    
    UT = hi*Dl_tilde*p5*(s2 - ph[0,0] - ph[0,1])*(U[0,1]-U[0,0]) + \
         p5*(jat[0,0] + jat[0,1])*nz
         
         
    # ============================
    # top edge flux (i, j-1/2)
    # ============================  
    phx = phipjm - phimjm
    phz = ph[0,0]-ph[0,-1]
    phn = np.sqrt(phx**s2 + phz**s2) 
    nz = divi(phz, phn, eps2, s0)
    
    #jat_jm = p5*(s1+(s1-k)*U[0,-1])*(s1-ph[0,-1]**s2)*psi_t[0,-1]      
    
    UB = hi*Dl_tilde*p5*(s2 - ph[0,0] - ph[0,-1])*(U[0,0]-U[0,-1]) + \
         p5*(jat[0,0] + jat[0,-1])*nz
    
    rhs_U = ( (UR-UL) + (UT-UB) ) * hi + sqrt2 * jat[0,0]
    tau_U = (s1+k) - (s1-k)*ph[0,0]
    
   
    return rhs_U/tau_U
    


@njit(parallel=True)
def rhs_psi(ps,ph,U,zz): return _rhs_psi(ps,ph,U,zz)


@njit(parallel=True)
def rhs_U(U,ph,psi_t): return _rhs_U(U,ph,psi_t)



def save_data(phi,U):
    
    c_tilde = ( s1+ (s1-k)*U )*( k*(s1+phi)/s2 + (s1-phi)/s2 )
    
    return np.vstack(( np.reshape(phi[1:-1,1:-1].T,(nv,1),order='F') , \
                      np.reshape(c_tilde[1:-1,1:-1].T,(nv,1),order='F') ) )


##############################################################################

psi0 = PARA.seed_initial(xx,lx,zz)
U0 = s0*psi0 + U_0

psi = set_halo(psi0.T)
U = set_halo(U0.T)
zz = set_halo(zz.T)


psi = set_BC(psi, 0, 1)
phi = np.tanh(psi/sqrt2)   # expensive replace
U =   set_BC(U, 0, 1)
Tishot[:,[0]] = save_data(phi,U)

psi6 = np.float64(psi)


#complie
start = time.time()
dPSI = rhs_psi(psi, phi, U, zz - R_tilde*t)
dU = rhs_U(U,phi,dPSI)

end = time.time()

print('elapsed: ', end - start )



start = time.time()




for jj in range(nts):

    for ii in range(int(Mt/nts)):
        
        #print(psi.dtype,phi.dtype,U.dtype,(zz - R_tilde*t).dtype)
        dPSI = rhs_psi(psi, phi, U, zz - R_tilde*t)
        
        
        jat = set_BC( p5*(s1+(s1-k)*U)*(s1-phi**s2)*dPSI , 0, 1)
        
        psi = psi + dt*dPSI
       
        U = U + dt*rhs_U(U, phi, jat )
       
        
        # add boundary
        psi = set_BC(psi, 0, 1)
    
        U = set_BC(U, 0, 1)
        
        phi = np.tanh(psi/sqrt2) 
        
        
        t += dt

    #print('now time is ',t)  
    Tishot[:,[jj+1]] = save_data(phi,U)

end = time.time()


print('elapsed: ', end - start )


Uf = U[1:-1,1:-1].T

#save(os.path.join(direc,filename),{'xx':xx*W0,'zz':zz[1:-1,1:-1].T*W0,'y':Tishot,'dt':dt*tau0,'nx':nx,'nz':nz,'t':t*tau0,'mach_time':end-start})
