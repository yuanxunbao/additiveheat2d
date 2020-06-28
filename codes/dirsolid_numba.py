#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:10:13 2020

@author: yigong qin, yuanxun bao
"""

import numba
from numba import njit, stencil, vectorize, float32, float64
import numpy as np
#import matplotlib.pyplot as plt

from dsinput_Takaki import phys_para, simu_para, IO_para 
import time

# print(numba.get_num_threads())

# numba.set_num_threads(2)

# print(numba.get_num_threads())

delta, k, lamd, R_tilde, Dl_tilde, lT_tilde, W0, tau0 = phys_para()
eps, alpha0, lxd, aratio, nx, dt, Mt, eta, filename = simu_para(W0,Dl_tilde)
nts, direc, r0 = IO_para(W0,lxd)

a_s = 1 - 3*delta
epsilon = 4*delta/a_s
a_12 = 4*a_s*epsilon
sqrt2 = np.sqrt(2.)

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
t=0

dxdz_in = 1./(dx*dz)  

hi= 1./dx

@njit
def set_halo(u):
    
    m,n = u.shape
    ub = np.zeros((m+2,n+2))
    
    ub[1:-1,1:-1] = u
    
    return ub

# the equations are divergence of fluxs, having the form F_x + J_z 
def initial(): # properties/parameters of interests
    
    r0 = 0.5625#*20
    r = np.sqrt( (xx-lx/2) **2+(zz)**2) - r0    
    psi0 = - r 
    U0 = 0*psi0 -0.3
    
    return psi0,U0


@vectorize([float32(float32, float32),
            float64(float64, float64)])
def atheta(ux, uz):
    
    ux2 = ux**2
    uz2 = uz**2
        
    # return MAG_sq2
    MAG_sq2 = (ux2 + uz2)**2
    
    if (MAG_sq2 > eps**2):
        
        return a_s*( 1 + epsilon*(ux2**2 + uz2**2) / MAG_sq2   )
        # return uz/MAG_sq2
    else:
        return a_s
    
    
@vectorize([float32(float32, float32),
            float64(float64, float64)])
def aptheta(ux, uz):
    
    ux2 = ux**2
    uz2 = uz**2
    
    MAG_sq2 = (ux2 + uz2)**2
    
    if (MAG_sq2 > eps**2):
        
        return -a_12*ux*uz*(ux2 - uz2) /  MAG_sq2
    
    else:
        return 0.0
    
    
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
        
        

@stencil
def _rhs_psi(ps,ph,U,zz,t):

    # ps = psi, ph = phi
    
    # =============================================================
    # 
    # 1. ANISOTROPIC DIFFUSION
    # 
    # =============================================================
    
    # these ps's are defined on cell centers
    psipjp=( ps[ 1, 1] + ps[ 0, 1] + ps[ 0, 0] + ps[ 1, 0] ) * 0.25
    psipjm=( ps[ 1, 0] + ps[ 0, 0] + ps[ 0,-1] + ps[ 1,-1] ) * 0.25
    psimjp=( ps[ 0, 1] + ps[-1, 1] + ps[-1, 0] + ps[ 0, 0] ) * 0.25
    psimjm=( ps[ 0, 0] + ps[-1, 0] + ps[-1,-1] + ps[ 0,-1] ) * 0.25
    
    phipjp=( ph[ 1, 1] + ph[ 0, 1] + ph[ 0, 0] + ph[ 1, 0] ) * 0.25
    phipjm=( ph[ 1, 0] + ph[ 0, 0] + ph[ 0,-1] + ph[ 1,-1] ) * 0.25
    phimjp=( ph[ 0, 1] + ph[-1, 1] + ph[-1, 0] + ph[ 0, 0] ) * 0.25
    phimjm=( ph[ 0, 0] + ph[-1, 0] + ph[-1,-1] + ph[ 0,-1] ) * 0.25
    
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
    phxn = ( ph[ 1, 0] - ph[-1, 0] ) * 0.5
    phzn = ( ph[ 0, 1] - ph[ 0,-1] ) * 0.5
    psxn = ( ps[ 1, 0] - ps[-1, 0] ) * 0.5
    pszn = ( ps[ 0, 1] - ps[ 0,-1] ) * 0.5
    
    A2 = atheta(phxn,phzn)**2
    gradps2 = (psxn)**2 + (pszn)**2
    extra =  -sqrt2 * A2 * ph[0,0] * gradps2
    

    # =============================================================
    # 
    # 3. double well (transformed): sqrt2 * phi + nonlinear terms
    # 
    # =============================================================
    
    Up = (zz[0,0] - R_tilde*t)/lT_tilde
    
    rhs_psi = ((JR-JL) + (JT-JB) + extra) * hi**2 + \
               sqrt2*ph[0,0] - lamd*(1-ph[0,0]**2)*sqrt2*(U[0,0] + Up) 
        
    
    # =============================================================
    # 
    # 4. dpsi/dt term
    # 
    # =============================================================
    tp = (1-(1-k)*Up)
    tau_psi = tp*A2 if tp >= k else k*A2
    
    return rhs_psi/tau_psi




@stencil
def _rhs_U(U,ph,psi_t):
    
    # define cell centered values
    phipjp=( ph[ 1, 1] + ph[ 0, 1] + ph[ 0, 0] + ph[ 1, 0] ) * 0.25
    phipjm=( ph[ 1, 0] + ph[ 0, 0] + ph[ 0,-1] + ph[ 1,-1] ) * 0.25
    phimjp=( ph[ 0, 1] + ph[-1, 1] + ph[-1, 0] + ph[ 0, 0] ) * 0.25
    phimjm=( ph[ 0, 0] + ph[-1, 0] + ph[-1,-1] + ph[ 0,-1] ) * 0.25

    # ============================
    # right edge flux (i+1/2, j)
    # ============================
    phx = ph[1,0]-ph[0,0]
    phz = phipjp - phipjm
    phn2 = phx**2 + phz**2
    nx = phx / np.sqrt(phn2) if (phn2 > eps) else 0.
    
    jat    = 0.5*(1+(1-k)*U[0,0])*(1-ph[0,0]**2)*psi_t[0,0]
    jat_ip = 0.5*(1+(1-k)*U[1,0])*(1-ph[1,0]**2)*psi_t[1,0]
        
    UR = Dl_tilde*0.5*(2 - ph[0,0] - ph[1,0])*(U[1,0]-U[0,0]) + \
         0.5*(jat + jat_ip)*nx/hi
         
         
    # ============================
    # left edge flux (i-1/2, j)
    # ============================
    phx = ph[0,0]-ph[-1,0]
    phz = phimjp - phimjm
    phn2 = phx**2 + phz**2
    nx = phx / np.sqrt(phn2) if (phn2 > eps) else 0.
    
    jat_im = 0.5*(1+(1-k)*U[-1,0])*(1-ph[-1,0]**2)*psi_t[-1,0]
    
    UL = Dl_tilde*0.5*(2 - ph[0,0] - ph[-1,0])*(U[0,0]-U[-1,0]) + \
         0.5*(jat + jat_im)*nx/hi
         
         
    # ============================
    # top edge flux (i, j+1/2)
    # ============================     
    phx = phipjp - phimjp
    phz = ph[0,1]-ph[0,0]
    phn2 = phx**2 + phz**2
    nz = phz / np.sqrt(phn2) if (phn2 > eps) else 0.
          
    jat_jp = 0.5*(1+(1-k)*U[0,1])*(1-ph[0,1]**2)*psi_t[0,1]      
    
    UT = Dl_tilde*0.5*(2 - ph[0,0] - ph[0,1])*(U[0,1]-U[0,0]) + \
         0.5*(jat + jat_jp)*nz/hi
         
         
    # ============================
    # top edge flux (i, j-1/2)
    # ============================  
    phx = phipjm - phimjm
    phz = ph[0,0]-ph[0,-1]
    phn2 = phx**2 + phz**2
    nz = phz / np.sqrt(phn2) if (phn2 > eps) else 0.
    
    jat_jm = 0.5*(1+(1-k)*U[0,-1])*(1-ph[0,-1]**2)*psi_t[0,-1]      
    
    UB = Dl_tilde*0.5*(2 - ph[0,0] - ph[0,-1])*(U[0,0]-U[0,-1]) + \
         0.5*(jat + jat_jm)*nz/hi 
    
    rhs_U = ( (UR-UL) + (UT-UB) ) * hi**2 + sqrt2 * jat
    tau_U = (1+k) - (1-k)*ph[0,0]
    
    
    return rhs_U/tau_U
    

@njit(parallel=True)
def rhs_psi(ps,ph,U,zz,t): return _rhs_psi(ps,ph,U,zz,t)


@njit(parallel=True)
def rhs_U(U,ph,psi_t): return _rhs_U(U,ph,psi_t)






##############################################################################

psi0,U0 = initial()
psi = set_halo(psi0.T)
U = set_halo(U0.T)
zz = set_halo(zz.T)


#phi0 = np.tanh(psi0/sqrt2) # expensive replace


start = time.time()

psi = set_BC(psi, 0, 1)
phi = np.tanh(psi/sqrt2)
U =   set_BC(U, 0, 1)


dPSI = rhs_psi(psi, phi, U, zz, t)
dPSI = set_BC(dPSI, 0, 1)
# dPSI_int=dPSI[1:-1,1:-1]


dU = rhs_U(U,phi,dPSI)

end = time.time()

print('elapsed: ', end - start )



start = time.time()

for i in range(10000):

    
    psi = set_BC(psi, 0, 1)

    U = set_BC(U, 0, 1)
    
    phi = np.tanh(psi/sqrt2) # expensive replace
    
    
    dPSI = rhs_psi(psi, phi, U, zz, t)
    dPSI = set_BC(dPSI, 0, 1)
    
    psi = psi + dt*dPSI
  
    U = U + dt*rhs_U(U,phi,dPSI)
    t += dt
    
end = time.time()


print('elapsed: ', end - start )

phif = np.tanh(psi[1:-1,1:-1].T/sqrt2)
Uf = U[1:-1,1:-1].T


