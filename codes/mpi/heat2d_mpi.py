
import numpy as np
from mpi4py import MPI
import scipy.io as sio



def initial_condition( x, y ):

    # sigma = 0.1       
    # u0 = np.exp(-(x-0.5)**2/(2*sigma**2))
    xx, yy = np.meshgrid(x,y)
    u0 = (xx - xx**2)*( yy - yy**2)
    return  u0



comm = MPI.COMM_WORLD		# initialize MPI
rank = comm.Get_rank()		# id of the current processor [0:nproc]
nproc = comm.Get_size()		# number of processors

nprocx = int(np.sqrt(nproc))
nprocy = int(nproc/nprocx)


px = rank%nprocx           # x id of current processor   [0:nprocx]
py = int(np.floor(rank/nprocx)) # y id of current processor  [0:nprocy]
 

print('total/x/y processors', nproc, nprocx, nprocy)

Lmin = 0 			# left domain of the whole problem
Lmax = 1.0			# right domain of the whole problem
Kappa = 1

npts = 16			# number of grid points in the current processor at one line 

Nx = nprocx*npts + 1    # total x grid points of domain
Ny = nprocy*npts + 1    # total y grid points of domain
Ntotal = Nx*Ny

dx = (Lmax-Lmin) / (Nx-1)

cfl = 0.4
dt = cfl * (dx**2/Kappa)

T = 0.025
Nt = int (T/dt)
dt = T/Nt



# root proc is responsible for initializing
if rank == 0:
    x = np.linspace(Lmin,Lmax,num=Nx)
    y = np.linspace(Lmin,Lmax,num=Ny)
    u0 = initial_condition( x, y )
    # print('initial cond',u0)
else:
    x = None
    y = None
    u0 = None

# assign grids for each processor

if px == nprocx-1:
    if py == nprocy-1:
        u_loc = np.zeros((npts+1,npts+1))
    else: u_loc = np.zeros((npts+1,npts))
    
elif px < nprocx-1 and py == nprocy-1:
    
    u_loc = np.zeros((npts,npts+1))    
    
elif px < nprocx-1 and py < nprocy-1:
    u_loc = np.zeros((npts,npts))
    
else: print('invalid proceesor ID occurs!')

if rank == 0:
    print('Root processor to scatter data')

# prepare to use Scatterv

sct=np.ones(nproc,dtype=int)*npts
sct[-1]=npts+1
dis=np.arange(0,Ntotal,npts)
dis=dis[:-1]
# print(sct)
# print(di)

# Root processor divides initial condition into chunkcs and send to all processors
comm.Scatterv([u0,sct,dis,MPI.DOUBLE],u_loc)


def update_heat(rank, nproc, dx, u_loc, Nt, dt, Kappa ):

    nx, ny = u_loc.shape    # local dimensions of each processor
    
    u = np.zeros((nx+2,ny+2))
    u[1:-1,1:-1] = u_loc
    unew = u

    # main loop 
    for k in range(Nt):
      
        # message 1, send LEFT to RIGHT
        if (px < nprocx-1):
            comm.send(u[nx,:], dest = rank+1, tag=1)


        if (0 < px):
            u[0,:] = comm.recv(source = rank-1, tag=1)

   
        # message 2, send from RIGHT to LEFT
        if (0 < px):
            comm.send(u[1,:], dest=rank-1, tag=2)


        if (px < nprocx-1):
            u[nx+1,:] = comm.recv(source=rank+1, tag=2)
            
        # message 3, send from bottom to top   
        
        if (py < nprocy-1):
            comm.send(u[:,ny], dest = rank+nprocx, tag=3)


        if (0 < py):
            u[:,0] = comm.recv(source = rank-nprocx, tag=3)         
        
        
        # message 4, send from top to bottom
        
        if (0 < py):
            comm.send(u[:,1], dest=rank-nprocx, tag=4)
        
        if (py < nprocy):
            u[:,ny+1] = comm.recv(source=rank+nprocx, tag=4)
            
            
            
        # set Neumann BCs
        if (px == 0): u[0,:] = u[2,:]

        if (px == nprocx-1): u[-1,:] = u[-3,:]
        
        if (py == 0): u[:,0] = u[:,2]
            
        if (py == nprocy-1): u[:,-1] = u[:,-3]


        unew[1:-1,1:-1] = u[1:-1,1:-1] + Kappa*dt/(dx**2)*( u[:-2,1:-1] + u[2:,1:-1] + u[1:-1,:-2] + u[1:-1,2:] - 4*u[1:-1,1:-1] )
        u = unew  

    return u

#u_sol = update_heat(rank, nproc, dx, u_loc, Nt, dt, Kappa)

#print(u_sol)
u_sol = u_loc

# Root processor gather all data 
comm.Barrier()
if rank == 0:
    print('Root processor to gather data')
    u_final = np.zeros((Nx, Ny))
else:
    u_final = None


#comm.Gatherv(u_sol[1:-1],[u_final,sct,dis,MPI.DOUBLE])
comm.Gatherv(u_sol,[u_final,sct,dis,MPI.DOUBLE])

if rank==0:
    sio.savemat('heat_final.mat',{'u_final':u_final,'u0':u0,'x':x,'y':y})



