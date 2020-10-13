#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:45:21 2020

@author: yigongqin
"""

# macro model/ with latent heat
# crank-nicolson/ center difference
# linear solver CG

import os
import numpy as np
from macro_param_low2 import phys_parameter, simu_parameter 
from scipy import sparse as sp
from scipy.sparse import linalg as spla
import matplotlib.pyplot as plt
import scipy.io as sio
from math import pi
import time
from scipy.optimize import fsolve
from scipy.interpolate import interp2d as itp2d
from scipy.interpolate import RectBivariateSpline as rbs

            
def sparse_cg(A, b, y, TOL, P, maxit):

      num_iters = 0

      def callback(xk):
         nonlocal num_iters
         num_iters+=1

      x,status = spla.cg(A, b, x0=y, tol=TOL, M=P, maxiter = maxit, callback=callback)
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
    gradT_x = np.ones((ny,nx)); gradT_y = np.ones((ny,nx)); 
    gradT_x[1:-1,1:-1] = ( Tn[1:-1,2:] - Tn[1:-1,:-2])/(2*h)
    gradT_y[1:-1,1:-1] = ( -Tn[2:,1:-1] + Tn[:-2,1:-1])/(2*h)
    gradT_n = np.sqrt( gradT_x**2 + gradT_y**2 )
    
    # set boundary condition for G
    
    #gradT_n[0,:] = 
    #gradT_n[-1,:] = gradT_n[-1,:]
    
    
    #grady_n = np.reshape(gradT_n,(nv,1),order='F') 
    
    #dTdt = ( y_new - y_old )/dt
    dTdt = ( Tn - T_old )/dt
    
    return gradT_n, - dTdt/gradT_n, gradT_x, gradT_y



def liquid_contour(Temp): # this is only for selecting initial points
    
    T = np.reshape(Temp,(ny,nx),order='F') 
    
    
    xj_arr = []
    yj_arr = []
    Tj_arr = []
    
    

    
    for ii in range(nx):
        jj = 0
        while(T[jj,ii]>=p.Tcon): jj +=1  
        
        if jj>0: jj = jj-1 # there was a bug here!!
        
        if yy[jj,0]< -10e-6/4 :
            #xj_arr = np.concatenate((xj_arr, xx[0,ii]))
            #yj_arr = np.concatenate((yj_arr, yy[jj,0]))
            #Tj_arr = np.concatenate((Tj_arr, T[jj,ii]))
            xj_arr.append(xx[0,ii])
            yj_arr.append(yy[jj,0])
            Tj_arr.append(T[jj,ii])
            #Tj_arr[ii] = T[jj,ii]   
        #Gj_arr[ii] = G[jj,ii]
        #Rj_arr[ii] = R[jj,ii]
        
    # compute thetaj_arr
    #thetaj_arr[0] = (yj_arr[1]-yj_arr[0])/dx
    #thetaj_arr[-1] = (yj_arr[-1]-yj_arr[-2])/dx
    #thetaj_arr[1:-1] = (yj_arr[2:]-yj_arr[:-2])/(2*dx)
    #thetaj_arr = np.arctan(thetaj_arr) 
    #Radj_arr = np.sqrt( (l0-xj_arr)**2+(yj_arr)**2 )    
    #thetaj_arr = np.arctan((l0-xj_arr)/yj_arr)*180/pi
    num_sam = len(xj_arr)
    
    Gj_arr = np.zeros(num_sam)
    Rj_arr = np.zeros(num_sam)
    betaj_arr = np.zeros(num_sam)
    
    return np.array(xj_arr), np.array(yj_arr), np.array(Tj_arr), Gj_arr, Rj_arr, betaj_arr



def get_initial_contour(Temp): # this is only for selecting initial points
    
    T = np.reshape(Temp,(ny,nx),order='F') 
    
    
    xj_arr = []
    yj_arr = []
    Tj_arr = []
    

    
    for ii in range(nx):
        jj = 0
        while(T[jj,ii]>=p.Tcon): jj +=1  
        
        # if jj>0: jj = jj-1 # there was a bug here!!
        # below 10e-6
        if yy[jj,0]< -10e-6:
            #xj_arr = np.concatenate((xj_arr, xx[0,ii]))
            #yj_arr = np.concatenate((yj_arr, yy[jj,0]))
            #Tj_arr = np.concatenate((Tj_arr, T[jj,ii]))
            xj_arr.append(xx[0,ii])
            yj_arr.append(yy[jj,0])
            Tj_arr.append(T[jj,ii])
            #Tj_arr[ii] = T[jj,ii]
        #Gj_arr[ii] = G[jj,ii]
        #Rj_arr[ii] = R[jj,ii]
        
    # compute thetaj_arr
    #thetaj_arr[0] = (yj_arr[1]-yj_arr[0])/dx
    #thetaj_arr[-1] = (yj_arr[-1]-yj_arr[-2])/dx
    #thetaj_arr[1:-1] = (yj_arr[2:]-yj_arr[:-2])/(2*dx)
    #thetaj_arr = np.arctan(thetaj_arr) 
    #Radj_arr = np.sqrt( (l0-xj_arr)**2+(yj_arr)**2 )    
    #thetaj_arr = np.arctan((l0-xj_arr)/yj_arr)*180/pi
    num_sam = len(xj_arr)
    
    
    Gj_arr = np.zeros(num_sam)
    Rj_arr = np.zeros(num_sam)
    betaj_arr = np.zeros(num_sam)
    
    return np.array(xj_arr), np.array(yj_arr), np.array(Tj_arr), Gj_arr, Rj_arr, betaj_arr


def QoI_save(t,xj_arr,yj_arr,Tj_arr,Gj_arr,Rj_arr, betaj_arr):
    

    # open one file and write the time, coordinates and G, R, theta
            
    depth = yj_arr[int((len(yj_arr)-1)/2)]    
    
    with open(s.outname,'w' if t==0 else 'a') as out_file:
        
        if t>t0 and depth < -40e-6:
            
            for iii in range(len(yj_arr)):
    
                    
                out_string = ''
                out_string += str('%5.3e'%t)
                out_string += ',' + str('%5.3e'%xj_arr[iii])
                out_string += ',' + str('%5.3e'%yj_arr[iii])
                out_string += ',' + str('%5.3e'%Tj_arr[iii])
                out_string += ',' + str('%5.3e'%Gj_arr[iii])
                out_string += ',' + str('%5.3e'%Rj_arr[iii])
                #out_string += ',' + str('%5.3e'%Radj_arr[iii])
                out_string += ',' + str('%5.3e'%betaj_arr[iii])
                out_string += '\n'
                out_file.write(out_string)
        
    
    
    return 





# calculate length of step
def stepj(ss, *arg):  
    # here gT_xn and gT_yn are interpolated and T_itp is interpolated temperature at next time step
    # here xj_n and yj_n are arrays
    xj_n, yj_n, T_itp, gT_xn, gT_yn = arg
    
    xj_np = xj_n + ss*gT_xn
    yj_np = yj_n + ss*gT_yn
    
    
    Tj_np = T_itp(xj_np, yj_np)
    
    
    return Tj_np - p.Tcon



def trajectory(xj_arrn, yj_arrn, gTx_n, gTy_n, R, y_n, y_np):  
    # calculate coordinates for time at n+1 and output interpered gradTx gradTy, R 
    
    T_n = np.reshape(y_n,(ny,nx),order='F') 
    T_np = np.reshape(y_np,(ny,nx),order='F') 
    
    num_sam = len(xj_arrn)
    
    
    Tj_arrn = np.zeros(num_sam)
    gTxj_arrn = np.zeros(num_sam)
    gTyj_arrn = np.zeros(num_sam)
    Rj_arrn = np.zeros(num_sam)
    sj_arr = np.zeros(num_sam)
    
    T_itp = itp2d( x, ycoor, T_np, kind = 'linear' ) # get interp object for next time step t_n+1

    
    gTxn_itp = itp2d( x, ycoor, gTx_n, kind = 'linear' ) 
    gTyn_itp = itp2d( x, ycoor, gTy_n, kind = 'linear' ) 
    R_itp = itp2d( x, ycoor, R, kind = 'linear' ) 
    Tn_itp = itp2d( x, ycoor, T_n, kind = 'linear' )
    
    for ii in range(num_sam):
        
        x_j = xj_arrn[ii];
        y_j = yj_arrn[ii]
        Tj_arrn[ii] = Tn_itp(x_j,y_j)
        gTxj_arrn[ii] = gTxn_itp(x_j,y_j)
        gTyj_arrn[ii] = gTyn_itp(x_j,y_j)
        Rj_arrn[ii] = R_itp(x_j,y_j)

        arg = ( x_j, y_j, T_itp, gTxj_arrn[ii], gTyj_arrn[ii] )
        sj_arr[ii] = fsolve( stepj, [0.0], args=arg)
    
    Gj_arrn = np.sqrt( gTxj_arrn**2 + gTyj_arrn**2 )
    betaj_arrn = np.arctan( gTyj_arrn/gTxj_arrn)*180/pi
    
    return  xj_arrn + sj_arr*gTxj_arrn, yj_arrn + sj_arr*gTyj_arrn,\
        xj_arrn, yj_arrn, Tj_arrn, Gj_arrn, betaj_arrn, Rj_arrn



'''
def trajectory(xj_arrn, yj_arrn, gTx_n, gTy_n, R, y_n, y_np):  
    # calculate coordinates for time at n+1 and output interpered gradTx gradTy, R 
    
    T_n = np.reshape(y_n,(ny,nx),order='F') 
    T_np = np.reshape(y_np,(ny,nx),order='F') 
    
    num_sam = len(xj_arrn)
    
    
    Tj_arrn = np.zeros(num_sam)
    gTxj_arrn = np.zeros(num_sam)
    gTyj_arrn = np.zeros(num_sam)
    Rj_arrn = np.zeros(num_sam)
    sj_arr = np.zeros(num_sam)
    
    T_itp = itp2d( x, ycoor, T_np, kind = 'cubic' ) # get interp object for next time step t_n+1

    
    gTxn_itp = itp2d( x_c, y_c, gTx_n, kind = 'cubic' ) 
    gTyn_itp = itp2d( x_c, y_c, gTy_n, kind = 'cubic' ) 
    R_itp = itp2d( x_c, y_c, R, kind = 'cubic' ) 
    Tn_itp = itp2d( x, ycoor, T_n, kind = 'cubic' )
    
    for ii in range(num_sam):
        
        x_j = xj_arrn[ii];
        y_j = yj_arrn[ii]
        Tj_arrn[ii] = Tn_itp(x_j,y_j)
        gTxj_arrn[ii] = gTxn_itp(x_j,y_j)
        gTyj_arrn[ii] = gTyn_itp(x_j,y_j)
        Rj_arrn[ii] = R_itp(x_j,y_j)

        arg = ( x_j, y_j, T_itp, gTxj_arrn[ii], gTyj_arrn[ii] )
        sj_arr[ii] = fsolve( stepj, [0.0], args=arg)
    
    Gj_arrn = np.sqrt( gTxj_arrn**2 + gTyj_arrn**2 )
    betaj_arrn = np.arctan( gTyj_arrn/gTxj_arrn)*180/pi
    
    return  xj_arrn + sj_arr*gTxj_arrn, yj_arrn + sj_arr*gTyj_arrn,\
        xj_arrn, yj_arrn, Tj_arrn, Gj_arrn, betaj_arrn, Rj_arrn
'''



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
ycoor = np.linspace(0,-ly,ny)

xx,yy = np.meshgrid(x,ycoor)


# centered grid
x_c = x[1:] - dx/2
y_c = ycoor[1:] - dx/2



dt = s.dt
CFL = p.alpha*dt/dx**2  #CFL number

lat = pi/2 * p.Lm/(p.Tl-p.Ts)  # non dimensionalized latent heat


l0 = lx/2 #laser starting point 


Mt= s.Mt
Tt = Mt*dt
t=0


# macro output sampling arrays



# set the power source to decay
Tt_arr = np.linspace(0, Tt-dt, Mt)

q0_arr = np.zeros(Mt)

t0 = s.t0

tstop = Tt



# ===================Sampling parameters=================
nts = s.nts
kts = int( Mt/nts )


#=====================Generate matrix========================

Lap,Q = sparse_laplacian(nx,ny)

I = sp.eye(nx*ny,format='csc')

A0 = Q @ (I - CFL * Lap)


# preconditioner
M2 = spla.splu(A0)
M_x = lambda x: M2.solve(x)
M = spla.LinearOperator((nv,nv), M_x)



##====================Initial condition=========================
# Temp = np.zeros((nv,nts+1))
# G_arr = np.zeros((nv,nts+1)); R_arr = np.zeros((nv,nts+1))
#T = np.zeros((ny,nx))      # temperature T0
#T = np.arange(nv).reshape((ny,nx))
T0 = p.Te*np.ones((ny,nx))
y = np.reshape(T0,(nv,),order='F')  #T0/y0
# Temp[:,[0]] = y

#Gv,Rv = gradients(y,y,dx)

#G_arr[:,[0]] = Gv; R_arr[:,[0]] = Rv

# y_l,A_l=imex_latent(y)
#plt.imshow(T0,cmap=plt.get_cmap('hot'))



'''
Tu = y[0:-1:ny]
U = Up_boundary(t,dx,Tu)          # U0

bU = np.zeros((nv,1))
bU[0:-1:ny] = U
'''


found_bottom = False



# qs_x spatial
qs_x = heat_source_x( x )

#======================time evolusion=======================

TOL = 1e-9 #cg tolerance
maxit=80

start = time.time()

for ii in range(Mt):
    
    A_l = dphase_func(y)
    
    A = A0 + lat*Q@A_l
    
    # obtain right hand side
    b = y + lat*A_l*y 
    
    T_top = y[0:-1:ny]
    set_top_bc(b, qs_x, T_top, t)
    
    
    ynew,stat,num_iter = sparse_cg(A, Q@b, y, TOL, M, maxit)
    
    
    
    # G,R,gradT_x, gradT_y = compute_GR(ynew, y, dx)
    
    G,R, gTx, gTy = gradients(ynew, y, dx)
    
    T_ii = np.reshape(ynew,(ny,nx), order='F')
    
    
    
    # keep track of the bottom of liquid interface
    idx_mid = int((nx+1)/2)
    Tmid = T_ii[:,idx_mid]
    idx_bottom = np.flatnonzero(Tmid > p.Tl)
    
    if idx_bottom.size == 0:
        idx_bottom = 0
    else:
        idx_bottom = idx_bottom[-1]
        
    
    if R[idx_bottom, idx_mid] > 0 and found_bottom == False and ii>1 : 
        # ii>1 is a hack because the deepest point bounce back at ii=1
        
        print('start sampling...')
        
        istart = ii;
        found_bottom = True
        xj_arr, yj_arr, Tj_arr, Gj_arr, Rj_arr, betaj_arr = liquid_contour(ynew) 
        # xj_arr, yj_arr, Tj_arr, Gj_arr, Rj_arr, betaj_arr = base_contour(R) 
        num_sam = len(xj_arr)
        num_time = Mt - istart 
        
        T_star = T_ii
        
        
        x_traj = np.zeros((num_time,num_sam))
        y_traj = np.zeros((num_time,num_sam))
        T_traj = np.zeros((num_time,num_sam))
        G_traj = np.zeros((num_time,num_sam))
        R_traj = np.zeros((num_time,num_sam))
        beta_traj =np.zeros((num_time,num_sam))
        time_traj = np.zeros(num_time)
        
        
    '''    
    if found_bottom == True:
        
        xj_arr, yj_arr, xj_old, yj_old, Tj_arr, Gj_arr, betaj_arr, Rj_arr \
            = trajectory(xj_arr, yj_arr, gradT_x, gradT_y, R, y, ynew)
        
        # QoI_save(t,xj_old,yj_old,Tj_arr,Gj_arr,Rj_arr, betaj_arr)
            
            
        x_traj[ii-istart,:] = xj_old
        y_traj[ii-istart,:] = yj_old
        T_traj[ii-istart,:] = Tj_arr    # is this Tj_arr defined on old?
        G_traj[ii-istart,:] = Gj_arr
        R_traj[ii-istart,:] = Rj_arr
        beta_traj[ii-istart,:] = betaj_arr
        time_traj[ii-istart] = (ii-istart)*dt;
    '''    
        
    
    if found_bottom == True:
        
        xj_arr, yj_arr, xj_old, yj_old, Tj_arr, Gj_arr, betaj_arr, Rj_arr \
            = trajectory(xj_arr, yj_arr, gTx, gTy, R, y, ynew)
        
        # QoI_save(t,xj_old,yj_old,Tj_arr,Gj_arr,Rj_arr, betaj_arr)
            
            
        x_traj[ii-istart,:] = xj_old
        y_traj[ii-istart,:] = yj_old
        T_traj[ii-istart,:] = Tj_arr    # is this Tj_arr defined on old?
        G_traj[ii-istart,:] = Gj_arr
        R_traj[ii-istart,:] = Rj_arr
        beta_traj[ii-istart,:] = betaj_arr
        time_traj[ii-istart] = (ii-istart)*dt;
    
    
    
    print( 'timestep = %d, number of iter = %d, stat = %d'%(ii, num_iter, stat) )
    print( 'bottom index = %d, R = %.3e, G = %.3e'%(idx_bottom, R[idx_bottom,idx_mid], G[idx_bottom,idx_mid] ) )
    
    
    
    # ynew=np.reshape(ynew,((nv,1)))
    
    # traj_filename = 'hello1.mat'
    # sio.savemat(os.path.join(s.direc, traj_filename), { 'A0':A0, 'A':A, 'bU':bU, 'b':b , 'qs':qs})   
        
            
    
    
    '''
    yuanxun: this loop plot how R>0, T>Ts, T>Tl interfaces evolve 
    '''
    
    
    if (ii+1)%20 == 0:
        
        '''
        fig, axes = plt.subplots(2,2)
        
        axes[0,0].imshow(  (R>0)  ,cmap='RdBu')
        axes[0,0].set_title('R>0')
        
        axes[0,1].imshow( np.reshape(ynew,(ny,nx),order='F') > p.Tl, cmap='hot')
        axes[0,1].set_title('T>Tl')
        
        axes[1,0].imshow( np.reshape(ynew,(ny,nx),order='F') > p.Ts, cmap='hot')
        axes[1,0].set_title('T>Ts')
        
        axes[1,1].imshow( np.reshape(ynew-y,(ny,nx),order='F') > 0, cmap='RdBu')
        axes[1,1].set_title('T dot')
        
        fig.suptitle('t = '+str(ii*dt))
        '''
        
        fig,axes = plt.subplots(2,2)
        pos = axes[0,0].imshow(G, cmap = 'RdBu')
        axes[0,0].set_title('G')
        fig.colorbar(pos, ax=axes[0,0])
        
        
        pos = axes[0,1].imshow(R, cmap = 'RdBu', vmin=0, vmax=0.01)
        axes[0,1].set_title('R')
        fig.colorbar(pos, ax=axes[0,1])
        
        
        pos = axes[1,0].imshow(-G*R, cmap = 'RdBu')
        axes[1,0].set_title('Tdot')
        fig.colorbar(pos, ax=axes[1,0])
        
        
        pos = axes[1,1].imshow( T_ii * (T_ii > p.Tl) , cmap = 'hot')
        axes[1,1].set_title('T > Tl')
        fig.colorbar(pos, ax=axes[1,1])
        
        
        
        
        
        
        '''
        fig, ax = plt.subplots()
        ax.imshow( np.reshape(ynew,(ny,nx),order='F') > p.Tl,  interpolation='bilinear', 
                  cmap='hot', extent=[x.min(), x.max(), ycoor.min(), ycoor.max()] )
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('time = {:.3f} s'.format(dt*(ii+1)))
        # fig.savefig('./movie/interface'+ '{:04d}'.format(int((ii+1)/5))+'.png', dpi=300, bbox_inches = "tight")
        '''
    
   
         
               
    
    y = ynew
    # latent heat
    # y_l,A_l=imex_latent(y)    
    # Up boundary
    # Tu = y[0:-1:ny]
    # U = Up_boundary(t,dx,Tu)   # U_n
    #plug in right hand
    # bU[0:-1:ny] = U
    
    t += dt

    
    
       #print('gmres iterations: ',counter)
Tf = np.reshape(y,(ny,nx),order='F')
# print(Tf[0,int(nx/4)],Tf[0,int(nx/2)])



'''
fig2 = plt.figure(figsize=[12,4])
ax2 = fig2.add_subplot(121)
plt.imshow(Tf,cmap=plt.get_cmap('hot'))
plt.colorbar()

ax3 = fig2.add_subplot(122)
ax3.plot(xj_arr,yj_arr)
ax3.set_aspect('equal')
ax3.set_ylim(-ly, 0)
ax3.set_xlim(0, lx)

fig3 = plt.figure(figsize=[12,4])
ax4 = fig3.add_subplot(121)
plt.imshow(G[1:-1,1:-1],cmap=plt.get_cmap('hot'))
plt.xlim()
plt.colorbar();plt.title('G')

ax5 = fig3.add_subplot(122)
ax5.plot(xj_arr,yj_arr)
plt.imshow(R[1:-1,1:-1],cmap=plt.get_cmap('hot'))
plt.colorbar();plt.title('R')

'''


#filename = 'Qlatdt' + str(dt)+'xgrids'+str(nx)+'.mat'
tempname = 'temp'+str(nx)


# sio.savemat(os.path.join(s.direc,s.filename),{'Tf':Tf, 'nx':nx,'ny':ny,'xx':xx,'yy':yy})


traj_filename = 'macro_output_ref.mat'
sio.savemat(os.path.join(s.direc, traj_filename), {'x_traj':x_traj,'y_traj':y_traj,'T_traj':T_traj,'G_traj':G_traj,'R_traj':R_traj,'beta_traj':beta_traj, 'time_traj':time_traj, 't_start':istart*dt, 'Tstar':T_star})




end =time.time()
print('time used: ',end-start)




# traj_filename = 'hello1.mat'
# sio.savemat(os.path.join(s.direc, traj_filename), { 'Tf':Tf, 'A0':A0, 'A':A, 'b':b})






