
import numpy as np
from mpi4py import MPI
import scipy.io as sio
import cupy as cp
from cupy.random import rand as cprand
import importlib
import sys
import os
from scipy.io import savemat as save
from numba import njit, cuda, vectorize, float64, float64, int32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
from numpy.random import random, rand
import time
import math




@njit
def set_halo(u):

    m,n = u.shape
    ub = np.zeros((m+ 2*ha_wd,n+ 2*ha_wd))

    ub[ha_wd:-ha_wd,ha_wd:-ha_wd] = u

    return ub

'''
-------------------------------------------------------------------------------------------------
CUDA KERNEL FUNCTIONS
-------------------------------------------------------------------------------------------------
'''

@cuda.jit
def heat(U,Unew):


    i,j = cuda.grid(2)
    m,n = U.shape

    if 0 < i < m-1 and 0 < j < n-1 :

       Unew[i,j] = U[i,j] + dt*Kappa/(dx*dy) * (U[i+1,j] +U[i,j+1] +U[i-1,j] +U[i,j-1] -4*U[i,j])
       
@cuda.jit
def setBC_gpu(ph, px, py, nprocx, nprocy, ha_wd):

    m,n = ph.shape
    i = cuda.grid(1)
    # no flux  at x = 0, lx
    if ( px==nprocx-1 and i<n ):
        ph[-ha_wd,i]   = ph[-ha_wd-2,i]
    if ( px==0 and i<n):
        ph[ha_wd-1,i]   = ph[ha_wd+1,i]
    # no flux  at z = 0, lz
    if ( py==0 and i<m):
        ph[i,ha_wd-1] = ph[i,ha_wd+1]
    if ( py==nprocy-1 and i<m):
        ph[i,-ha_wd] = ph[i,-ha_wd-2]

@cuda.jit    
def BC_421(ph, ph_BC): # size of ph is nx+2*ny+2, ph_BC is 2(nx+ny), ha_wd

    m,n = ph.shape
    nx = m-2*ha_wd ;ny = n-2*ha_wd 
    i,j = cuda.grid(2)
    # the order: [0+hw:2*hw,hw:-hw],[-2hw:-hw:,hw:-hw:],[hw:-hw,hw:2hw],[hw:-hw,-2hw:-hw]

    if (  i< ny and j < ha_wd ):
        ph_BC[i,j] = ph[j+ha_wd,i+ha_wd]
        ph_BC[i+ny,j] = ph[j+nx,i+ha_wd]
    if (  i < nx and j < ha_wd ):
        ph_BC[i+2*ny,j] = ph[i+ha_wd,j+ha_wd]
        ph_BC[i+2*ny+nx,j] = ph[i+ha_wd,j+ny]

@cuda.jit        
def BC_124(ph, ph_BC): # size of ph is nx+2*ny+2, nbh_BC is 2(nx+ny), ha_wd

    m,n = ph.shape
    nx = m-2*ha_wd ;ny = n-2*ha_wd
    i,j = cuda.grid(2)
    # the order: [0,:],[-1,:],[:,0],[:,-1]

    if (  i< ny and j < ha_wd ):
        ph[j,i+ha_wd] = ph_BC[i,j]
        ph[j+nx+ha_wd,i+ha_wd] = ph_BC[i+ny,j] 
    if (  i < nx and j < ha_wd ):
        ph[i+ha_wd,j] = ph_BC[i+2*ny,j]
        ph[i+ha_wd,j+ny+ha_wd] = ph_BC[i+2*ny+nx,j]       
        
   # the order of recv is [0,0], [-1,0], [0,-1], [-1,-1]
    
    if i < ha_wd and j < ha_wd:
      ph[i,j]  =     ph_BC[i+2*ny+2*nx+0*ha_wd,j];   
      ph[i+nx+ha_wd,j] =   ph_BC[i+2*ny+2*nx+1*ha_wd,j];   
      ph[i,j+ny+ha_wd] =   ph_BC[i+2*ny+2*nx+2*ha_wd,j];   
      ph[i+nx+ha_wd,j+ny+ha_wd] = ph_BC[i+2*ny+2*nx+3*ha_wd,j];  



'''
-------------------------------------------------------------------------------------------------
MPI ENVIRONMENT SETTING
-------------------------------------------------------------------------------------------------
'''
def BC_comm(sendbuf, recvbuf, nx ,ny):

    if ( px < nprocx-1 ):
           # right_send = phi_gpu[-2*ha_wd:-ha_wd,:].reshape((ha_wd,ny+2*ha_wd))
            comm.Send([sendbuf[ny:2*ny,:],MPI.DOUBLE], dest = rank+1, tag=1)

    if ( px > 0 ):
            comm.Recv([recvbuf[:ny,:],MPI.DOUBLE],source = rank-1, tag=1)
           # phi_gpu[:ha_wd,:] = left_recv
    # sending direction:  X decreases
    if ( px > 0 ):
            comm.Send([sendbuf[:ny,:],MPI.DOUBLE], dest = rank-1, tag=2)
    if ( px < nprocx-1 ):
            comm.Recv([recvbuf[ny:2*ny,:],MPI.DOUBLE],source = rank+1, tag=2)
    
    # sending direction:  Y(Z) increases
    if ( py < nprocy-1 ):
            comm.Send([sendbuf[2*ny+nx:2*ny+2*nx,:],MPI.DOUBLE], dest = rank+nprocx, tag=3)
    if ( py>0 ):
            comm.Recv([recvbuf[2*ny:2*ny+nx,:],MPI.DOUBLE],source = rank-nprocx, tag=3)
    # sending direction:  Y(Z) decreases
    if ( py >0 ):
            comm.Send([sendbuf[2*ny:2*ny+nx,:],MPI.DOUBLE], dest = rank-nprocx, tag=4)
    if ( py < nprocy -1 ):
            comm.Recv([recvbuf[2*ny+nx:2*ny+2*nx,:],MPI.DOUBLE],source = rank +nprocx, tag=4)
    


        # the order: [0,:],[-1,:],[:,0],[:,-1]
        # the order of recv is [0,0], [-1,0], [0,-1], [-1,-1] send the corner data from [:,0] [:,-1]
    if ( px < nprocx-1 and py < nprocy-1):
            comm.Send([sendbuf[2*ny+2*nx-ha_wd:2*ny+2*nx,:],MPI.DOUBLE], dest = rank+1+nprocx, tag=5)

    if ( px > 0 and py > 0 ):
            comm.Recv([recvbuf[-4*ha_wd:-3*ha_wd,:],MPI.DOUBLE],source = rank-1-nprocx, tag=5)
    # sending direction:  X decreases    send [0,0] recv [-1,-1]
    if ( px > 0 and py > 0):
            comm.Send([sendbuf[2*ny:2*ny+ha_wd,:],MPI.DOUBLE], dest = rank-1-nprocx, tag=6)
    if ( px < nprocx-1 and py < nprocy-1 ):
            comm.Recv([recvbuf[-ha_wd:,:],MPI.DOUBLE],source = rank+1+nprocx, tag=6)

    # sending direction:  Y(Z) increases  send [0,-1] recv [-1,0]  
    if ( py < nprocy-1 and px > 0 ):
            comm.Send([sendbuf[2*ny+nx:2*ny+nx+ha_wd,:],MPI.DOUBLE], dest = rank+nprocx-1, tag=7)
    if ( py>0 and px < nprocx-1 ):
            comm.Recv([recvbuf[-3*ha_wd:-2*ha_wd,:],MPI.DOUBLE],source = rank-nprocx+1, tag=7)
    # sending direction:  Y(Z) decreases  send [-1,0] recv [0,-1]
    if ( py>0 and px < nprocx-1):
            comm.Send([sendbuf[2*ny+nx-ha_wd:2*ny+nx,:],MPI.DOUBLE], dest = rank-nprocx+1, tag=8)
    if ( py < nprocy -1 and px > 0):
            comm.Recv([recvbuf[-2*ha_wd:-ha_wd,:],MPI.DOUBLE],source = rank +nprocx-1, tag=8)


    return 



comm = MPI.COMM_WORLD		# initialize MPI
rank = comm.Get_rank()		# id of the current processor [0:nproc]
nproc = comm.Get_size()		# number of processors

num_gpus_per_node = 4
gpu_name = cuda.select_device( rank % num_gpus_per_node)

if rank == 0: print('GPUs on this node', cuda.gpus)
print('device id',gpu_name,'host id',rank )

nprocx = int(np.ceil(np.sqrt(nproc)))
nprocy = int(np.ceil(nproc/nprocx))


px = rank%nprocx           # x id of current processor   [0:nprocx]
py = int(np.floor(rank/nprocx)) # y id of current processor  [0:nprocy]
print('px ',px,'py ',py,' for rank ',rank) 

if rank ==0: print('total/x/y processors', nproc, nprocx, nprocy)


'''
-------------------------------------------------------------------------------------------------
CAST PARAMETERS/ INITIAL CONDITION ALLOCATION ON CPU AND GPU
-------------------------------------------------------------------------------------------------
'''
# set initial condition on each CPU and allocate the data on GPU

Lmin = float64(0) 			# left domain of the whole problem
Lmax = float64(1.0)			# right domain of the whole problem
Kappa = float64(1.0)

npts = int32(100)			# number of grid points in the current processor at one line 
Nx = nprocx*npts+1 #rank    # total x grid points of domain
Ny = nprocy*npts+1 #rank    # total y grid points of domain
Ntotal = Nx*Ny

len_block = (Lmax-Lmin)/nprocx 

dx = (Lmax-Lmin) / (Nx-1)
dx = float64(dx)
dy = float64(dx)
print(dx,dy)
cfl = 0.2
dt = cfl * (dx**2/Kappa)
dt = float64(dt)

T = float64(0.01)
Mt = int(T/dt)

lminx = px*len_block
lminy = py*len_block
# assign grids for each processor

if px == 0:
    if py == 0:
        nx = npts+1; ny=npts+1;
        x = np.linspace(lminx,lminx+len_block,num=npts+1)
        y = np.linspace(lminy,lminy+len_block,num=npts+1)
    else:    
        nx = npts+1; ny=npts;
        x = np.linspace(lminx,lminx+len_block,num=npts+1)
        y = np.linspace(lminy+dy,lminy+len_block,num=npts)

elif px > 0 and py == 0:
    nx = npts; ny=npts+1;
    x = np.linspace(lminx+dx,lminx+len_block,num=npts)
    y = np.linspace(lminy,lminy+len_block,num=npts+1)

elif px > 0 and py > 0:
    nx = npts; ny=npts;
    x = np.linspace(lminx+dx,lminx+len_block,num=npts)
    y = np.linspace(lminy+dy,lminy+len_block,num=npts)

else: print('invalid proceesor ID occurs!')
yy,xx = np.meshgrid(y,x)
phi = (xx-xx**2 )*( yy-yy**2 )
#phi = (2+np.arange(nx*ny)).reshape((nx,ny))
phi = phi.astype(np.float64)
#print('I am rank ', rank, 'my initial condition is: x ',x,'y',y )

ha_wd = 10   #!!!!!!!!!!!!!!!!!! for now just assume ha_wd is an even number!!!!!!!!!!!!!!!!!!!!
phi = set_halo(phi)
#print('rank',rank,phi[ha_wd:-ha_wd,ha_wd:-ha_wd])
print('rank',rank,'dimension',phi.shape,'\n')
phi_gpu = cuda.to_device(phi)
phi_swp = cuda.device_array_like(phi_gpu)
BCsend = cuda.device_array([2*nx+2*ny,ha_wd],dtype=np.float64);
BCrecv = cuda.device_array([2*nx+2*ny+4*ha_wd,ha_wd],dtype=np.float64)
'''
-------------------------------------------------------------------------------------------------
GPU CALCULATIONS
-------------------------------------------------------------------------------------------------
'''
# CUDA kernel invocation parameters
# cuda 2d grid parameters
tpb2d = (16,16)
bpg_x = math.ceil(phi.shape[0] / tpb2d[0])
bpg_y = math.ceil(phi.shape[1] / tpb2d[1])
bpg2d = (bpg_x, bpg_y)
# cuda 1d grid parameters
tpb = 16 * 1
bpg = math.ceil( np.max( [phi.shape[0], phi.shape[1]] ) / tpb )
#print('2d threads per block: ({0:2d},{1:2d})'.format(tpb2d[0], tpb2d[1]))
print('2d blocks per grid: ({0:2d},{1:2d})'.format(bpg2d[0], bpg2d[1]))
print('(threads per block, block per grid 1d) = ({0:2d},{1:2d})'.format(tpb, bpg))

bpg2dBC = (2*(bpg_x+bpg_y),math.ceil(ha_wd/tpb) )

start = time.time()
BC_421[bpg2dBC,tpb2d](phi_gpu, BCsend)
#if rank == 0: print('sendbuf',BCsend.copy_to_host())
BC_comm(BCsend, BCrecv, nx ,ny)
BC_124[bpg2dBC,tpb2d](phi_gpu, BCrecv)
setBC_gpu[bpg,tpb](phi_gpu, px, py, nprocx, nprocy, ha_wd)



#setBC_gpu[bpg,tpb](phi_gpu, px, py, nprocx, nprocy)
#BC_comm(phi_gpu,ha_wd)
for kt in range(int(Mt/2)):

      heat[bpg2d, tpb2d](phi_gpu,phi_swp)
     # BC_421[bpg2dBC,tpb2d](phi_swp, BCsend)
     # comm.Barrier()
     # BC_comm(BCsend, BCrecv, nx ,ny)
     # comm.Barrier()
     # BC_124[bpg2dBC,tpb2d](phi_swp, BCrecv)
      setBC_gpu[bpg,tpb](phi_swp, px, py, nprocx, nprocy, ha_wd)
   
      heat[bpg2d, tpb2d](phi_swp,phi_gpu)
      if (2*kt)%ha_wd==0:
           BC_421[bpg2dBC,tpb2d](phi_gpu, BCsend)
     # comm.Barrier()
           BC_comm(BCsend, BCrecv, nx ,ny)
     # comm.Barrier()
           BC_124[bpg2dBC,tpb2d](phi_gpu, BCrecv)
      setBC_gpu[bpg,tpb](phi_gpu, px, py, nprocx, nprocy, ha_wd)
 
    
end = time.time()
print('I am rank',rank,'time used: ', end-start)
phi_out = phi_gpu.copy_to_host()
if rank == 0: print('rank',rank,phi_out[ha_wd:-ha_wd,ha_wd:-ha_wd])
send = BCsend.copy_to_host()
recv = BCrecv.copy_to_host()
#print('rank',rank,'send',send,'recv',recv)
save('heat2dmultil'+'rank'+str(rank)+'nx'+str(nx) + 'ha_wd' + str(ha_wd) + '.mat',{'T':phi_out[ha_wd:-ha_wd,ha_wd:-ha_wd]})

'''
-------------------------------------------------------------------------------------------------
GATHER DATA AND OUTPUT
-------------------------------------------------------------------------------------------------
'''

