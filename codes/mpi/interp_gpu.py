
import numpy as np
import scipy.io as sio
from scipy import linalg as la
import importlib
import sys
import os
from scipy.io import savemat as save
from numba import njit, cuda, vectorize, float64, float64, int32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
from numpy.random import random, rand
import time
import math
from math import pi


'''
-------------------------------------------------------------------------------------------------
CUDA KERNEL FUNCTIONS
-------------------------------------------------------------------------------------------------
'''

@cuda.jit
def XYT_lin_interp(x, y, t, X, Y, T,  u_3d,u_2d,u_m,v_3d,v_2d,v_m ):
    # array description: x (nx,); y (ny,); t (scalar); X (Nx,); Y (Ny,); T (Nt,); u_3d (Nx, Ny, Nt); u_2d (Nx,Ny)
    # note that u_m, v_m here have halo, so dimension is (nx+2, ny+2)
    i,j = cuda.grid(2)
    Nx,Ny,Nt = u_3d.shape
    m, n = u_m.shape  #(m = nx+2, n = ny +2)
    nx = m -2; ny = n -2;
    # first step is get the id of the time kt and kt+1, do a interpation between kt layer and kt+1 layer first 
    Dt = T[1] - T[0]
    kt = int( ( t - T[0] )/Dt )
    delta_t = ( t - T[0] )/Dt - kt        

   # if i< Nx and j < Ny:
   #      u_2d[i,j] = (1-delta_t)*u_3d[i,j,kt] + delta_t*u_3d[i,j,kt+1]    
   #      v_2d[i,j] = (1-delta_t)*v_3d[i,j,kt] + delta_t*v_3d[i,j,kt+1]
    # then preform a standard 2d interplation from Nx, Ny to nx+2, ny+2
   # cuda.syncthreads() 
    Dx = X[1] - X[0]  # for now assume uniform mesh

    if  i < nx and j < ny :
         
        kx = int( ( x[i] - X[0] )/Dx )
        delta_x = ( x[i] - X[0] )/Dx - kx

        ky = int( ( y[j] - Y[0] )/Dx )
        delta_y = ( y[j] - Y[0] )/Dx - ky
         
        u_m[i+1,j+1] = ( (1-delta_x)*(1-delta_y)*u_3d[kx,ky,kt] + (1-delta_x)*delta_y*u_3d[kx,ky+1,kt] \
                     +delta_x*(1-delta_y)*u_3d[kx+1,ky,kt] +   delta_x*delta_y*u_3d[kx+1,ky+1,kt] )*(1-delta_t) + \
                   ( (1-delta_x)*(1-delta_y)*u_3d[kx,ky,kt+1] + (1-delta_x)*delta_y*u_3d[kx,ky+1,kt+1] \
                     +delta_x*(1-delta_y)*u_3d[kx+1,ky,kt+1] +   delta_x*delta_y*u_3d[kx+1,ky+1,kt+1] )*delta_t 


      #  v_m[i,j] = (1-delta_x)*(1-delta_y)*v_2d[kx,ky] + (1-delta_x)*delta_y*v_2d[kx,ky+1] \
     #                +delta_x*(1-delta_y)*v_2d[kx+1,ky] +   delta_x*delta_y*v_2d[kx+1,ky+1]          
   
    return 

@cuda.jit
def XYT_lin_interp2(x, y, t, X, Y, T,  u_3d,u_2d,u_m,v_3d,v_2d,v_m ):
    # array description: x (nx,); y (ny,); t (scalar); X (Nx,); Y (Ny,); T (Nt,); u_3d (Nx, Ny, Nt); u_2d (Nx,Ny)
    # note that u_m, v_m here have halo, so dimension is (nx+2, ny+2)
    i,j = cuda.grid(2)
    Nx,Ny,Nt = u_3d.shape
    m, n = u_m.shape  #(m = nx+2, n = ny +2)
    nx = m -2; ny = n -2;
    # first step is get the id of the time kt and kt+1, do a interpation between kt layer and kt+1 layer first
    #Dt = T[1] - T[0]
    #kt = int( ( t - T[0] )/Dt )
    #delta_t = ( t - T[0] )/Dt - kt

    #if i< Nx and j < Ny:
    #     u_2d[i,j] = (1-delta_t)*u_3d[i,j,kt] + delta_t*u_3d[i,j,kt+1]
    #     v_2d[i,j] = (1-delta_t)*v_3d[i,j,kt] + delta_t*v_3d[i,j,kt+1]
    # then preform a standard 2d interplation from Nx, Ny to nx+2, ny+2
   # cuda.syncthreads()
    Dx = X[1] - X[0]  # for now assume uniform mesh

    if 0 < i < nx+1 and 0 < j < ny+1 :

        kx = int( ( x[i-1] - X[0] )/Dx )
        delta_x = ( x[i-1] - X[0] )/Dx - kx

        ky = int( ( y[j-1] - Y[0] )/Dx )
        delta_y = ( y[j-1] - Y[0] )/Dx - ky

        u_m[i,j] = (1-delta_x)*(1-delta_y)*u_2d[kx,ky] + (1-delta_x)*delta_y*u_2d[kx,ky+1] \
                     +delta_x*(1-delta_y)*u_2d[kx+1,ky] +   delta_x*delta_y*u_2d[kx+1,ky+1]

        v_m[i,j] = (1-delta_x)*(1-delta_y)*v_2d[kx,ky] + (1-delta_x)*delta_y*v_2d[kx,ky+1] \
                     +delta_x*(1-delta_y)*v_2d[kx+1,ky] +   delta_x*delta_y*v_2d[kx+1,ky+1]

    return       


        
# set initial condition on each CPU and allocate the data on GPU

lx = 2*pi
ly = 2*pi
nx = 1000
ny = 1000
dt = 0.0001
tend = 2*pi
Mt = int(tend/dt) 
x = np.linspace(0,lx,nx)
y = np.linspace(0,ly,ny)

xx, yy = np.meshgrid(x,y,indexing='ij')


T_true = np.cos(xx)*np.cos(yy)*np.cos(Mt*dt)
print('corner value:', T_true[0,0])
# insert macro data here

ref = 2
Nx = 5*ref
Ny = 5*ref
Nt = 10
Tend = 2*pi
X = np.linspace(0,lx,Nx)
Y = np.linspace(0,ly,Ny)
mac_t = np.linspace(0,Tend,Nt)

XX,YY,TT = np.meshgrid(X,Y,mac_t, indexing='ij')

T_trueB = np.cos(XX[:,:,0])*np.cos(YY[:,:,0])*np.cos(Mt*dt)



T_3D = np.cos(XX)*np.cos(YY)*np.cos(TT)
alpha_3D = np.cos(XX)*np.cos(YY)*np.cos(TT)

T_3D_gpu = cuda.to_device(T_3D)
alpha_3D_gpu = cuda.to_device(alpha_3D)


T_m = cuda.device_array((nx+2,ny+2))
alpha_m = cuda.device_array((nx+2,ny+2))

T_2D_gpu = cuda.device_array((Nx,Ny))
alpha_2D_gpu = cuda.device_array((Nx,Ny))

'''
-------------------------------------------------------------------------------------------------
GPU CALCULATIONS
-------------------------------------------------------------------------------------------------
'''
# CUDA kernel invocation parameters
# cuda 2d grid parameters
tpb2d = (16,16)
bpg_x = math.ceil( (nx+2) / tpb2d[0])
bpg_y = math.ceil( (ny+2) / tpb2d[1])
bpg2d = (bpg_x, bpg_y)
# cuda 1d grid parameters
tpb = 16 * 1
bpg = math.ceil( np.max( [ nx +2 , ny + 2] ) / tpb )
#print('2d threads per block: ({0:2d},{1:2d})'.format(tpb2d[0], tpb2d[1]))
print('2d blocks per grid: ({0:2d},{1:2d})'.format(bpg2d[0], bpg2d[1]))
print('(threads per block, block per grid 1d) = ({0:2d},{1:2d})'.format(tpb, bpg))
start = time.time()

#setBC_gpu[bpg,tpb](phi_gpu, px, py, nprocx, nprocy)
#BC_comm(phi_gpu,ha_wd)
for kt in range(Mt):
        
      t = (kt+1)*dt 
      XYT_lin_interp[bpg2d, tpb2d](x, y, t, X, Y, mac_t, T_3D_gpu, T_2D_gpu, T_m, alpha_3D_gpu, alpha_2D_gpu,alpha_m ) 
   #   XYT_lin_interp2[bpg2d, tpb2d](x, y, t, X, Y, mac_t, T_3D_gpu, T_2D_gpu, T_m, alpha_3D_gpu, alpha_2D_gpu,alpha_m ) 
end = time.time()
print('time used: ', end-start)
T2d = T_2D_gpu.copy_to_host()
a2d = alpha_2D_gpu.copy_to_host()
print('after time interp: ', T2d)
T_out = T_m.copy_to_host()
alpha_out = alpha_m.copy_to_host()
print('final temperature',T_out)
err1d = la.norm(T2d-T_trueB)/la.norm(T_trueB)
err2d = la.norm(T_out[1:-1,1:-1]-T_true)/la.norm(T_true)
erra1d = la.norm(a2d-T_trueB)/la.norm(T_trueB)
erra2d = la.norm(alpha_out[1:-1,1:-1]-T_true)/la.norm(T_true)
print('erre we got: 1d, 2d',err1d,err2d,erra1d, erra2d )
#print('rank',rank,'send',send,'recv',recv)
#save('heat2dnoB'+'rank'+str(rank)+'nx'+str(nx)+'.mat',{'T':phi_out[1:-1,1:-1]})

