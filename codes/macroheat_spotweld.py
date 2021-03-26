#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:04:23 2020

@author: yuanxun
"""

import importlib
import os, sys
import numpy as np
from macro_input2 import phys_parameter, simu_parameter 
from scipy import sparse as sp
from scipy.sparse import linalg as spla
import matplotlib.pyplot as plt
import scipy.io as sio
from math import pi
import time
from scipy.optimize import fsolve, brentq

from scipy.interpolate import interp2d as itp2d
from scipy.interpolate import interp1d 
from scipy.interpolate import griddata


def sparse_cg(A, b, u0, TOL, P, maxit):

      num_iters = 0

      def callback(xk):
         nonlocal num_iters
         num_iters+=1

      x,status = spla.cg(A, b, x0=u0, tol=TOL, M=P, maxiter = maxit, callback=callback)
      return x,status,num_iters
  

def sparse_laplacian(nx,ny):
    
    # Neumann BC Laplacian
    Ix = sp.eye(nx,format='csc'); Iy = sp.eye(ny,format='csc')
    Ix[0,0] = 0.5; Ix[-1,-1] = 0.5
    Iy[0,0] = 0.5; Iy[-1,-1] = 0.5
    


    Dxx = sp.diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx)).toarray()
    Dxx[0,1]=2; Dxx[-1,-2]=2

    Dyy = sp.diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny)).toarray()
    Dyy[0,1]=2; Dyy[-1,-2]=2


    # L = sp.kronsum(Dyy,Dxx,format='csc')
    L = sp.kronsum(Dxx,Dyy,format='csc')

    Q = sp.kron(Ix,Iy,format='csc')
    # Q = sp.kron(Ix,Iy,format='csc')
    
    return L,Q


def gradients2(y, dTdt, h):
    # yn is new result, yp is previous time step
    Tn = np.reshape(y, (nx,ny), order='F')
    dTdt2d = np.reshape(dTdt, (nx,ny), order='F')
    
    #gradT_n = np.ones((ny,nx))11
    # gradT_x = np.ones((ny-1,nx-1)); gradT_y = np.ones((ny-1,nx-1)); 
    gradT_x =  ( Tn[2:,  1:-1] - Tn[:-2, 1:-1]) / (2*h)
    gradT_y =  ( Tn[1:-1,  2:] - Tn[1:-1, :-2]) / (2*h)
    gradT_n = np.sqrt( gradT_x**2 + gradT_y**2 )
    
    # set boundary condition for G
    
    #gradT_n[0,:] = 
    #gradT_n[-1,:] = gradT_n[-1,:]
    
    
    #grady_n = np.reshape(gradT_n,(nv,1),order='F') 
    
    #dTdt = ( y_new - y_old )/dt
    
    return gradT_n, -dTdt2d[1:-1,1:-1]/gradT_n, gradT_x, gradT_y


def convert_dim(x,y,u):
    
    
    xd = x * phys.len_scale * 1e6   # [um]
    yd = y * phys.len_scale * 1e6   # [um]
    Temp = u*(phys.Tl-phys.Ts)+phys.Ts   # [K]
    
    return xd,yd,Temp


def len_dim(x): return x *  phys.len_scale * 1e6     # [um]

def temp_dim(u): return u*(phys.Tl-phys.Ts) + phys.Ts  # [K]

def heat_dim(u): return u*(phys.Tl-phys.Ts)  # [K]

def vel_dim(v): return v * phys.len_scale / phys.t_scale # [m/s]

def time_dim(t): return t * phys.t_scale    #[s]

def gradT_dim(g): return heat_dim(g)/ phys.len_scale

def dT_dim(dT): return heat_dim(dT)/ phys.t_scale




def heat_source( x,t, x0 ):
    
    return phys.n1 * np.exp( -2*(x-x0 - phys.Vs*phys.rb/phys.kappa*t)**2 )

    # return p.q0 * np.exp( -2*(x-l0)**2 / p.rb**2 ) * np.exp( -2*(p.Vs*t)**2 / p.rb**2 )


def dphase_trans(u):
    
    mask = (u > 0 ) & (u < 1)
    
    T_lat = pi/2. * np.sin(pi * u) * mask;
    
    return sp.diags( T_lat, format = 'csc')


def set_top_bc(rhs, qs, u_top, t):
    
    # qs_t = heat_source_t(t
    
    # bU = - 2*dx/p.K*( -qs + p.hc*(T_top-p.Te) + p.epsilon*p.sigma*(T_top**4-p.Te**4) ) 
    
    bU = -2*simu.h * ( -qs + phys.n2*(u_top + phys.n3) + phys.n4*((u_top + phys.n6)**4 - phys.n5**4  ) )
    
    # rhs[0:-1:simu.ny] =  rhs[0:-1:simu.ny] + bU * CFL 
    rhs[-simu.nx:] = rhs[-simu.nx:] + bU * CFL 
    
    # return qs, bU
    
    
    
def get_initial_contour_T(u_interp, G_interp, R_interp, center , theta ):
    
    x0=center[0]
    y0=center[1]
    
    
    # print(x0,y0)
    # u2d = np.reshape(u, (ny,nx), order='F')
    
    # u_interp = itp2d(x,y, u2d)
    
    # R_interp = itp2d(x_i, y_i, R)
    # G_interp = itp2d(x_i, y_i, G)
    
    
    X = np.zeros(theta.size)
    Y = np.zeros(theta.size)
    Gt= np.zeros(theta.size)
    Rt= np.zeros(theta.size)
    Tt= np.zeros(theta.size)
    
    
    for j in range(theta.size):
        
        
        f = lambda s : u_interp( x0 + s*np.cos(theta[j]) , y0 + s*np.sin(theta[j]) ) - 1.0
        
        sj = brentq(f, y.min() ,0)
        
        # print(sj)
        
        X[j] = x0 + sj *np.cos(theta[j])
        Y[j] = y0 + sj *np.sin(theta[j])
        
        Gt[j] = G_interp(X[j], Y[j])
        Rt[j] = R_interp(X[j], Y[j])
        Tt[j] = u_interp(X[j], Y[j])
        
    
    return X, Y, Gt, Rt, Tt


def track_sl_interface(x0,y0,gx,gy,unew_interp, Gnew_interp, Rnew_interp):
    
    vx = gx(x0,y0)
    vy = gy(x0,y0)
    
    vn = np.sqrt(vx**2+vy**2)
    vx = vx/vn
    vy = vy/vn

    f = lambda s: u_interp( x0+s*vx, y0+s*vy ) - 1
    
    s0 = fsolve(f, [0.0])

    # print(s0)

    x1 = x0 + s0*vx
    y1 = y0 + s0*vy
    
    return x1,y1, Gnew_interp(x1,y1), Rnew_interp(x1,y1)
    
    
    
def comp_pool_depth(u_interp, center):
    
    f = lambda s : u_interp(  center[0] , center[1] + s*np.sin(np.pi/2) ) - 1
                            
    pd = brentq(f, y.min() ,0)
    
    # x_bottom[0] = center[0]
    # x_bottom[1] = center[1] + sj * np.sin(np.pi/2)    
    
    # print( u_interp(x_bottom[0], x_bottom[1]) )
    
    
    return np.abs(pd) 
    
    
def export_mat(CFL):
    
    Lap,Q = sparse_laplacian(simu.nx, simu.ny)
    I = sp.eye(simu.nx * simu.ny,format='csc')
    
    A0 = Q @ ( I - CFL*Lap)

    # preconditioner
    M2 = spla.splu(A0)
    M_x = lambda x: M2.solve(x)
    M = spla.LinearOperator((nv,nv), M_x)

    return Lap, Q, I, A0, M


def comp_dist2top_v(X,Y,gx,gy):

    d2t = np.zeros((X.size))
    
    for j in range(X.size):
        
        x0 = X[j]
        y0 = Y[j]
    
        s = - y0 / gy(x0,y0)
        x1 = x0 + s * gx(x0,y0)
    
        d2t[j] = np.sqrt(  (x1-x0)**2 + (0-y0)**2  )  
        
    return d2t



def comp_dist2top_p(x0,y0,gx,gy):

    s = - y0 / gy(x0,y0)
    x1 = x0 + s * gx(x0,y0)
    
    return np.sqrt(  (x1-x0)**2 + (0-y0)**2  )  
        


if len(sys.argv) == 4:
    
    phys = phys_parameter( float(sys.argv[1]),\
                           float(sys.argv[2]),\
                           float(sys.argv[3]))
 
        
    simu = simu_parameter(phys)
    
    h  = simu.h
    nx = simu.nx
    ny = simu.ny
    dt = simu.dt
    dt_off = simu.dt_off
    
else:  
    phys = phys_parameter( 1800, 100e-3, 3e-4 )
    simu = simu_parameter( phys )
    
    h  = simu.h
    nx = simu.nx
    ny = simu.ny
    dt = simu.dt
    dt_off = simu.dt_off
    
    




#=============================================================================
#
#   COMPUTATIONAL GRID    
#
#=============================================================================
    
print('dimensionless domain lx x ly: {:.2f} x {:.2f}'.format(simu.lx, simu.ly))
print('grid size nx x ny: {:d} x {:d}'.format(simu.nx, simu.ny))
print('mesh width h: {:.4f}'.format( simu.h ))

nv = simu.ny * simu.nx #number of DOFs
nv_int = (simu.ny-2)*(simu.nx-2) # number of interior DOFs

x = np.linspace(-simu.lx/2,simu.lx/2, simu.nx)
y = np.linspace(-simu.ly,0,simu.ny)

yy,xx = np.meshgrid(y,x)


# centered grid
x_c = x[1:] - simu.h/2
y_c = y[1:] - simu.h/2

# interior point
x_i = x[1:-1]
y_i = y[1:-1]



'''
# grid for dns
lw = 2e-3
n_dns = 200
x_dns = np.linspace(lx/2 - lw, lx/2 + lw, num=2*n_dns+1, endpoint =True )
y_dns = np.linspace(0,-lw, num=n_dns+1, endpoint = True)
nv_dns = x_dns.size * y_dns.size
xx_dns, yy_dns = np.meshgrid(x_dns,y_dns)
'''



#=============================================================================
#
#   PRECOMPUTE MATRICES
#
#=============================================================================

CFL = simu.dt/ simu.h**2  #CFL number
inv_Ste = 1./phys.Ste
Lap, Q, I, A0, M = export_mat(CFL)



# initial condition
u = phys.u0 * np.ones(nv)


# This loop start from t=0 and stop when the deepest S-L interface is reached


iter_melt = 0
t=0
pool_depth = y.max()

nplot=1
dTdt = 0

start = time.time()


#==============================================================================
#
#     MELTING, SPOT ON
#     
#==============================================================================

# while (reach_bottom == False and iter_melt < Mt):
while iter_melt < simu.Mt_spot_on :
    
    A_l = dphase_trans(u)
    
    A = A0 + inv_Ste*Q@A_l
    
    # obtain right hand side
    # b = u + lat*A_l*u  
    b = u + inv_Ste*A_l*u 
    
    
    # u_top = u[0:-1:simu.ny]
    u_top = u[-simu.nx:]
    
        
    qs = heat_source(x,t*0, simu.source_x[0])
    

    set_top_bc(b, qs, u_top, t)
    
    
    unew,stat,num_iter = sparse_cg(A, Q@b, u, simu.cg_tol, M, simu.maxit)
    
    dTdt_new = np.abs((unew - u)/simu.dt) 
    
    u2d = np.reshape(unew, (simu.nx, simu.ny), order='F')
    u_interp = itp2d(x,y,u2d.T)
    
    
    
    try:
        
        pd = comp_pool_depth(u_interp, simu.source_x)

        print( 'cur depth', len_dim(pd)  )
            
    except ValueError:
        
        print('Temperature is below T_liquid')
    
    
    
    u = unew
    dTdt = dTdt_new
    t += simu.dt
    iter_melt += 1

    
    print(t)
    
    
    if iter_melt%10 == 0:
    
        
        xd,yd,ud = convert_dim(xx, yy, u2d)

        fig, ax = plt.subplots()
        h1 = ax.pcolormesh( xd,yd,ud, cmap = 'hot')
        ax.contour(xd,yd,ud, [phys.Ts, phys.Tl])
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.set_ylim( len_dim(-3),0)
        ax.set_xlim( len_dim(-3), len_dim(3) )
        fig.colorbar(h1)
    
    

    
cur_depth = pd



sys.exit()


#==============================================================================
#
#     SPOT OFF
#     
#==============================================================================
    

theta = np.linspace(1,179,179) * pi/180
ntraj = theta.size
X_arr = np.zeros((ntraj, simu.sol_max))
Y_arr = np.zeros((ntraj, simu.sol_max))
G_arr = np.zeros((ntraj, simu.sol_max))
R_arr = np.zeros((ntraj,  simu.sol_max))
dlen_arr = np.zeros((ntraj, simu.sol_max))  # distance from s-l interface (arc length)
d2t_arr = np.zeros((ntraj,  simu.sol_max))   # approxiamte distance to top boundary


X_sl = np.zeros((ntraj, simu.sol_max))
Y_sl = np.zeros((ntraj, simu.sol_max))



CFL = simu.dt_off/ simu.h**2  #CFL number
Lap, Q, I, A0, M = export_mat(CFL)

all_reach_top = False
reach_bottom = False
iter_sol = 0
t=0
iter_max_depth=0
reach_top = np.zeros((theta.size, 400),dtype = bool)


G, R, grad_x, grad_y = gradients2(u, dTdt, h)
u_interp = itp2d(x,y,u2d.T)            
R_interp  = itp2d(x_i, y_i, R.T)
G_interp  = itp2d(x_i, y_i, G.T)
gx_interp = itp2d(x_i, y_i, grad_x.T)
gy_interp = itp2d(x_i, y_i, grad_y.T)


# sys.exit()



while all_reach_top == False & iter_sol < simu.sol_max : 
    
# while iter_sol < 80:
    
    
    A_l = dphase_trans(u)
    
    A = A0 + inv_Ste*Q@A_l
    
    # obtain right hand side
    # b = u + lat*A_l*u  
    b = u + inv_Ste*A_l*u 
    
    
    # u_top = u[0:-1:simu.ny]
    u_top = u[-simu.nx:]
    
        
    qs = heat_source(x,t*0, simu.source_x[0]) * (1 - t/simu.t_tapper) 
    
    print('t/tm', t/simu.t_tapper)
    

    set_top_bc(b, qs, u_top, t)
    
    
    unew,stat,num_iter = sparse_cg(A, Q@b, u, simu.cg_tol, M, simu.maxit)
    
    dTdt_new = (unew - u)/simu.dt_off 
    
    
    G, R, grad_x, grad_y = gradients2(unew, dTdt_new, h)
    
    u2d = np.reshape(unew, (simu.nx, simu.ny), order='F')
    
    
    unew_interp = itp2d(x,y,u2d.T)
    Rnew_interp  = itp2d(x_i, y_i, R.T)
    Gnew_interp  = itp2d(x_i, y_i, G.T)
    gxnew_interp = itp2d(x_i, y_i, grad_x.T)
    gynew_interp = itp2d(x_i, y_i, grad_y.T)


    try:
        
        
        
        pd = comp_pool_depth(unew_interp, simu.source_x)

        if pd > cur_depth :
            
            print( 'still melting, cur depth is %.6f, iter = %d\n'%(len_dim(pd), iter_sol)  )
        
        elif reach_bottom:
            
            
            print( 'solidificaition, cur depth is %.6f, iter= %d\n'%(len_dim(pd), iter_sol)  )
            
            
            kk = iter_sol-iter_max_depth
            
            
            X_sl[:,kk],Y_sl[:,kk], gg, rr,Tt =\
                get_initial_contour_T(unew_interp, Gnew_interp, Rnew_interp, simu.source_x, theta ) 
            
            
            
            for jj in range(theta.size):
        
                if reach_top[jj,iter_sol] == False :  
            
                
                    X_arr[jj,kk], Y_arr[jj,kk], G_arr[jj,kk], R_arr[jj,kk]  = \
                            track_sl_interface(X_arr[jj,kk-1], Y_arr[jj,kk-1], gx_interp, gy_interp, unew_interp, Gnew_interp, Rnew_interp)
                        
                    # d2c_arr[jj,iter_sol+1] = np.sqrt( (X_arr[jj,iter_sol+1]-center[0])**2 + (Y_arr[jj,iter_sol+1]-center[1])**2 )
                    dlen_arr[jj,kk] = dlen_arr[jj,kk-1] + np.sqrt(  (X_arr[jj,kk]-X_arr[jj,kk-1])**2 + (Y_arr[jj,kk]-Y_arr[jj,kk-1])**2  )
                    
                    
                    # alpha_arr[jj,iter_sol+1] = compute_growth_dir(X_arr[jj,iter_sol+1], Y_arr[jj,iter_sol+1], gx_new, gy_new)
                    
                    
                    d2t_arr[jj, kk] = comp_dist2top_p(X_arr[jj,kk], Y_arr[jj,kk], gxnew_interp, gynew_interp)
                
                    if d2t_arr[jj,kk] > simu.near_top : 
                    # if Y_arr[jj,iter_sol+1]+dx  < y.max() : 
                         reach_top[jj,kk] = False
                    else :
                         reach_top[jj,kk:] = True
                        
                         X_arr[jj,kk+1:] = X_arr[jj,kk]
                         Y_arr[jj,kk+1:] = Y_arr[jj,kk]
                         d2t_arr[jj,kk+1:] = d2t_arr[jj,kk]
                         dlen_arr[jj,kk+1:] = dlen_arr[jj,kk]
                        
                        
                else:
                    
                    T_min = unew_interp(X_arr[jj,iter_sol+1], X_arr[jj,iter_sol+1])
                    #T_max = unew_interp(Xtop_arr[jj], Ytop_arr[jj])
                    
                    # print(T_min)
                    # print(T_max)
                    
                    
                    
            if len(np.flatnonzero( ~reach_top[:,kk])) == 0: 
                
                
                all_reach_top = True
                iter_reach_top = kk
            
            
            
            
            
            
        else:
            
            max_depth = cur_depth
            iter_max_depth = iter_sol
            
            reach_bottom = True
            print( 'bottom is reached, cur depth is %.6f, iter = %d\n'%(len_dim(pd), iter_sol) )
            
            
            '''
            G, R, grad_x, grad_y = gradients2(u, dTdt, h)
            
            R_interp  = itp2d(x_i, y_i, R.T)
            G_interp  = itp2d(x_i, y_i, G.T)
            gx_interp = itp2d(x_i, y_i, grad_x.T)
            gy_interp = itp2d(x_i, y_i, grad_y.T)
            '''
            
            # initial normal
            X_arr[:,0],Y_arr[:,0], G_arr[:,0], R_arr[:,0],Tt =\
                get_initial_contour_T(unew_interp, Gnew_interp, Rnew_interp, simu.source_x, theta ) 
                
            # initial interface    
            X_sl[:,0] = X_arr[:,0]
            Y_sl[:,0] = Y_arr[:,0]
                
                
                
            # predicted distance to top 
            d2t_arr[:,0] = comp_dist2top_v(X_arr[:,0], Y_arr[:,0], gxnew_interp, gynew_interp)
                
                
                
            
    except ValueError:
        
        print('An exception is raised, iter %d, pd = %.6f'%(iter_sol, len_dim(pd)))
        
        
        
    
    
    
    cur_depth = pd
    dTdt = dTdt_new
    u = unew
    t += simu.dt_off
    iter_sol += 1
    
    
    u_interp = unew_interp
    G_interp = Gnew_interp
    R_interp = Rnew_interp
    gx_interp = gxnew_interp
    gy_interp = gynew_interp
    
    print(t)
    
    
    if iter_sol%10 == 0:
    
        
        xd,yd,ud = convert_dim(xx, yy, u2d)

        fig, ax = plt.subplots()
        h1 = ax.pcolormesh( xd,yd,ud, cmap = 'hot')
        ax.contour(xd,yd,ud, [phys.Ts, phys.Tl])
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.set_ylim( len_dim(-2),0)
        ax.set_xlim( len_dim(-2), len_dim(2) )
        fig.colorbar(h1)
        
        
        
        
# once finished, resize time
        
        

u2dnew = temp_dim(u2d)
Rnew = vel_dim(R)
Gnew = gradT_dim(G)
dTnew = dT_dim( np.reshape(dTdt, (nx,ny), order = 'F') )


Rnew_arr = vel_dim( R_arr )
Gnew_arr = gradT_dim(G_arr)



'''
X_arr = X_arr[:, :iter_reach_top]
Y_arr = Y_arr[:, :iter_reach_top]
X_sl = X_sl[:, :iter_reach_top]  
Y_sl = Y_sl[:, :iter_reach_top]

             
G_arr = G_arr[:, :iter_reach_top]
R_arr = R_arr[:, :iter_reach_top]
dlen_arr = dlen_arr[:, :iter_reach_top]
d2t_arr = d2t_arr[:, :iter_reach_top]         

    
fig2,ax2 = plt.subplots()
ax2.plot(X_sl[0:-1:3,0:-1:3], Y_sl[0:-1:3,0:-1:3], color = 'r')
ax2.plot(X_arr[0:-1:3,0:-1:10].T, Y_arr[0:-1:3,0:-1:10].T, color='b')
ax2.axis('equal')
'''