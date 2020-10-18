#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:45:21 2020

@author: yigongqin, yuanxun

"""

# macro model/ with latent heat
# crank-nicolson/ center difference
# linear solver CG


import os, sys
import numpy as np
from macro_param_low2 import phys_parameter, simu_parameter 
from scipy import sparse as sp
from scipy.sparse import linalg as spla
import matplotlib.pyplot as plt
import scipy.io as sio
from math import pi
import time
from scipy.optimize import fsolve, brentq

from scipy.interpolate import interp2d as itp2d

            
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


    L = sp.kronsum(Dyy,Dxx,format='csc')

    Q = sp.kron(Ix,Iy,format='csc')
    
    return L,Q
    
def heat_source_x( x ):
    
    return p.q0 * np.exp( -2*(x-l0)**2 / p.rb**2 )


def heat_source_t( t ):
    
    return np.exp( -2*(p.Vs*t)**2 / p.rb**2 )


def dphase_func(T):
    
    mask = (T >= p.Ts) & (T <= p.Tl)
    
    T_lat = np.sin(pi * (T-p.Ts)/(p.Tl-p.Ts)) * mask;
    
    return sp.diags( T_lat, format = 'csc')


def set_top_bc(rhs, qs_x, T_top, t):
    
    qs_t = heat_source_t(t)
    
    qs = qs_t * qs_x
    
    bU = - 2*dx/p.K*( -qs + p.hc*(T_top-p.Te) + p.epsilon*p.sigma*(T_top**4-p.Te**4) ) 
    
    rhs[0:-1:ny] =  rhs[0:-1:ny] + bU * CFL 
    # return qs, bU



def compute_GR(y, y_old, h):
    
    T     = np.reshape(y,(ny,nx),order='F')
    T_old = np.reshape(y_old,(ny,nx),order='F')
    
    gradT_x = ( T[:,1:]-T[:,:-1] ) / h
    gradT_x_c = ( gradT_x[1:,:] + gradT_x[:-1,:] )/2  # defined on all cell centers
    
    gradT_y = ( T[1:,:]-T[:-1,:] ) / h
    gradT_y_c = ( gradT_y[:,1:] + gradT_y[:,:-1] )/2  # defined on all cell centers
    
    G = np.sqrt( gradT_x_c**2 + gradT_y_c**2)
    
    Tdot = (T - T_old)/dt
    
    Tdot_e = ( Tdot[1:,:]+Tdot[:-1,:])/2
    Tdot_c = (Tdot_e[:,1:]+Tdot_e[:,:-1])/2
    
    
    R = -Tdot_c / G 
    
    return G, R,  gradT_x_c, gradT_y_c



def gradients(y_new,y_old,h):
    # yn is new result, yp is previous time step
    Tn = np.reshape(y_new,(ny,nx),order='F')
    T_old = np.reshape(y_old,(ny,nx),order='F')
    
    #gradT_n = np.ones((ny,nx))
    gradT_x = np.ones((ny-1,nx-1)); gradT_y = np.ones((ny-1,nx-1)); 
    gradT_x = ( Tn[1:-1,2:] - Tn[1:-1,:-2])/(2*h)
    gradT_y = ( -Tn[2:,1:-1] + Tn[:-2,1:-1])/(2*h)
    gradT_n = np.sqrt( gradT_x**2 + gradT_y**2 )
    
    # set boundary condition for G
    
    #gradT_n[0,:] = 
    #gradT_n[-1,:] = gradT_n[-1,:]
    
    
    #grady_n = np.reshape(gradT_n,(nv,1),order='F') 
    
    #dTdt = ( y_new - y_old )/dt
    dTdt = ( Tn - T_old )/dt
    
    return gradT_n, -dTdt[1:-1,1:-1]/gradT_n, gradT_x, gradT_y



def get_initial_contour_T(u_interp, G_interp, R_interp, center , theta ):
    
    x0=center[0]
    y0=center[1]
    
    
    print(x0,y0)
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
        
        
        f = lambda s : u_interp( x0 + s*np.cos(theta[j]) , y0 + s*np.sin(theta[j]) ) - p.Tl
        
        sj = brentq(f, y.min() ,0)
        
        print(sj)
        
        X[j] = x0 + sj *np.cos(theta[j])
        Y[j] = y0 + sj *np.sin(theta[j])
        
        Gt[j] = G_interp(X[j], Y[j])
        Rt[j] = R_interp(X[j], Y[j])
        Tt[j] = u_interp(X[j], Y[j])
        
    
    return X, Y, Gt, Rt, Tt



def get_initial_contour_R(u_interp, G_interp, R_interp, center , theta ):
    
    x0=center[0]
    y0=center[1]
    
    
    print(x0,y0)
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
        
        
        f = lambda s : R_interp( x0 + s*np.cos(theta[j]) , y0 + s*np.sin(theta[j]) )
        
        sj = brentq(f, y.min() ,0)
        
        print(sj)
        
        X[j] = x0 + sj *np.cos(theta[j])
        Y[j] = y0 + sj *np.sin(theta[j])
        
        Gt[j] = G_interp(X[j], Y[j])
        Rt[j] = R_interp(X[j], Y[j])
        Tt[j] = u_interp(X[j], Y[j])
    
    return X, Y, Gt, Rt, Tt



def reach_bottom_check(u,R):
    
    
    # keep track of the bottom of liquid interface
    u2d = np.reshape(u,(ny,nx),order='F')
    idx_mid = int((nx+1)/2)
    
    
    # find liquid index along the mid line of the melt pool
    liquid_set = np.flatnonzero( u2d[:,idx_mid] > p.Tl )
    
    
    if liquid_set.size == 0:
        idx_bottom = 0
    else:
        idx_bottom = liquid_set[-1]+1
        
        
    print( 'bottom index = %d, R = %.3e'%(idx_bottom, R[idx_bottom,idx_mid]) )

        
    if R[idx_bottom, idx_mid] > 0 and u.max() > p.Tl : return True
    else: return False


def track_sl_interface(x0,y0,gx,gy,u_interp,G_interp,R_interp):
    
    vx = gx(x0,y0)
    vy = gy(x0,y0)
    
    vn = np.sqrt(vx**2+vy**2)
    vx = vx/vn
    vy = vy/vn

    f = lambda s: u_interp( x0+s*vx, y0+s*vy ) - p.Tl
    
    s0 = fsolve(f, [0.0])

    # print(s0)

    x1 = x0 + s0*vx
    y1 = y0 + s0*vy
    
    return x1,y1, G_interp(x1,y1), R_interp(x1,y1)

#====================parameters==========================

p = phys_parameter()
s = simu_parameter(p.rb)

lx = s.lxd
aratio = s.asp_ratio
ly = lx*aratio


## make xx, yy, the dimensions and index 

nx = s.nx
dx = lx/(nx-1)
print('mesh width is', dx)

ny = int((nx-1)*aratio+1)
print('y grids ny:',ny)
#nx= N+1  #x grids;ny= M+1  #y grids
nv= ny*nx #number of variables

x = np.linspace(0,lx,nx)
y = np.linspace(0,-ly,ny)
ycoor = y

xx,yy = np.meshgrid(x,y)


# centered grid
x_c = x[1:] - dx/2
y_c = y[1:] - dx/2

# interior point
x_i = x[1:-1]
y_i = y[1:-1]



dt = s.dt
CFL = p.alpha*dt/dx**2  #CFL number

lat = pi/2 * p.Lm/(p.Tl-p.Ts)  # non dimensionalized latent heat constant


l0 = lx/2 #laser starting point 


Mt= s.Mt
t_max = Mt*dt
t=0



#=====================Generate matrix=========================================

Lap,Q = sparse_laplacian(nx,ny)

I = sp.eye(nx*ny,format='csc')

A0 = Q @ (I - CFL * Lap)


# preconditioner
M2 = spla.splu(A0)
M_x = lambda x: M2.solve(x)
M = spla.LinearOperator((nv,nv), M_x)
#=====================================================================


reach_bottom = False # flag when melting reach the max radius
all_reach_top = False # flag when all sampling trajectories reach top 

# qs_x spatial
qs_x = heat_source_x( x )




theta = np.linspace(1,90,90) * pi/180
Ntraj = theta.size



# initial condition
u = p.Te * np.ones(nv)


# This loop start from t=0 and stop when the deepest S-L interface is reached

TOL = 1e-9 #cg tolerance
maxit=80
iter_melt = 0
iter_sol = 0
t=0


#==============================================================================
#
#     MELTING
#     
#==============================================================================


start = time.time()

while (reach_bottom == False and iter_melt < Mt):
    
    A_l = dphase_func(u)
    
    A = A0 + lat*Q@A_l
    
    # obtain right hand side
    b = u + lat*A_l*u 
    
    
    u_top = u[0:-1:ny]
    set_top_bc(b, qs_x, u_top, t)
    
    
    unew,stat,num_iter = sparse_cg(A, Q@b, u, TOL, M, maxit)
    
    
    dTdt = np.abs((unew - u)/dt) 
    print('max cooling rate = ', dTdt.max())

    
    # G,R, grad_x, grad_y = compute_GR(unew, u, dx)
    G,R, grad_x, grad_y = gradients(unew, u, dx)
    
    
    u2d = np.reshape(unew, (ny,nx), order='F')
    u_interp = itp2d(x,y,u2d)
    R_interp = itp2d(x_i, y_i, R)
    G_interp = itp2d(x_i, y_i, G)
    gx = itp2d(x_i, y_i, grad_x)
    gy = itp2d(x_i, y_i, grad_y)

    
    
    print('iter = %d'%(iter_melt))
    
    if reach_bottom_check(unew, R) :
        
        
        print('reach the bottom at %d', iter_melt+1)
        reach_bottom = True  
        
        # sample base countour
        # X, Y, Gt, Rt = get_initial_contour(unew, G, R) 
        
        
        X,Y, Gt, Rt,Tt = get_initial_contour_T(u_interp, G_interp, R_interp, [l0,0], theta ) 
        # p.Tl = Tt.mean()
        
        # X2,Y2, Gt2, Rt2,Tt2 = get_initial_contour3(u_interp, G_interp, R_interp, [lx/2,0], theta ) 
        
        
        
    u = unew
    t += dt
    iter_melt += 1

    print( 'timestep = %d, number of iter = %d, stat = %d'%( iter_melt , num_iter, stat) )
    
    
    
T_sl_start = np.reshape(unew,(ny,nx),order='F')
 
    


if reach_bottom == False:
    
    print( 'Initial S-L interface not found, heat source not strong enough'  )
    sys.exit()


#==============================================================================
#
#     SOLIDIFICATION 
#     
#==============================================================================
    
    
n_sol = Mt - iter_melt
iter_sol
t_sol_start = t    
t = 0 # rest time


center = np.array([lx/2,0])
time_traj=np.arange(n_sol+1)*dt
X_arr = np.zeros((X.size, n_sol+1))
Y_arr = np.zeros((X.size, n_sol+1))
G_arr = np.zeros((X.size, n_sol+1))
R_arr = np.zeros((X.size, n_sol+1))
d2c_arr = np.zeros((X.size, n_sol+1))

# False if not hit top, true if hit top
reach_top = np.zeros((X.size, n_sol+1),dtype = bool)

X_arr[:,0] = X
Y_arr[:,0] = Y
G_arr[:,0] = Gt
R_arr[:,0] = Rt # we assume the starting interface has zero velocity
d2c_arr[:,0] = np.sqrt( (X-center[0])**2 + (Y-center[1])**2 )



while all_reach_top == False and iter_sol < n_sol :
    # terminate either all trajectories reach top or max iter is reached
    
    
    
    A_l = dphase_func(u)
    
    A = A0 + lat*Q@A_l
    
    # obtain right hand side
    
    b = u + lat*A_l*u 
    
    
    u_top = u[0:-1:ny]
    set_top_bc(b, qs_x, u_top, t + t_sol_start)
    
    
    unew,stat,num_iter = sparse_cg(A, Q@b, u, TOL, M, maxit)
    
    # G,R, grad_x, grad_y = compute_GR(unew, u, dx)
    G,R, grad_x, grad_y = gradients(unew, u, dx)
    
    
    
    u2d = np.reshape(unew, (ny,nx), order='F')
    u_interp = itp2d(x,y,u2d)
    R_interp = itp2d(x_i, y_i, R)
    G_interp = itp2d(x_i, y_i, G)
    gx = itp2d(x_i, y_i, grad_x)
    gy = itp2d(x_i, y_i, grad_y)
    
    
    
    for jj in range(X.size):
        
        if reach_top[jj,iter_sol] == False :  
    
        
            X_arr[jj,iter_sol+1], Y_arr[jj,iter_sol+1], G_arr[jj,iter_sol+1], R_arr[jj,iter_sol+1]  = \
                    track_sl_interface(X_arr[jj,iter_sol], Y_arr[jj,iter_sol], gx, gy, u_interp, G_interp, R_interp)
                
            d2c_arr[jj,iter_sol+1] = np.sqrt( (X_arr[jj,iter_sol+1]-center[0])**2 + (Y_arr[jj,iter_sol+1]-center[1])**2 )        
        
            if Y_arr[jj,iter_sol+1]+dx  < y.max() : 
                reach_top[jj,iter_sol+1] = False
            else :
                reach_top[jj,iter_sol+1:] = True
        
    dTdt = np.abs((unew - u)/dt) 
    
    print( 'timestep = %d, number of iter = %d, stat = %d'%( iter_sol , num_iter, stat) )    
    
    
    # check if all trajectories reach the top, stop the simulation. 
    if np.sum(reach_top[:,iter_sol]) == Ntraj : all_reach_top = True
    
    u = unew
    t += dt
    iter_sol+= 1
    
end = time.time()

print('elapsed time = %.2f'%(end-start))


   
traj_filename = "macroheat_Q%dW_Vs%.1fmmps_rb%.1fmm_%dx%d.mat_dt%.2e.mat"%(p.Q, p.Vs*1000, p.rb*1000, nx, ny, dt)

sio.savemat(os.path.join(s.direc, traj_filename), \
            {'x_traj':X_arr[:,:iter_sol-1] ,'y_traj': Y_arr[:,:iter_sol-1] ,\
             'G_traj':G_arr[:,:iter_sol-1],'R_traj':R_arr[:,:iter_sol-1],\
             'time_traj':time_traj[:iter_sol-1], 't_start': t_sol_start, \
             'reach_top':reach_top[:,:iter_sol-1],\
             'theta':theta, 'distance2center':d2c_arr[:,:iter_sol-1],\
             'h':dx, 'dt':dt,\
             'T_sl_start':T_sl_start})
    