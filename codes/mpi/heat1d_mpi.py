
import numpy as np
from mpi4py import MPI
import scipy.io as sio



def initial_condition( x ):

    sigma = 0.1
        
    u0 = np.exp(-(x-0.5)**2/(2*sigma**2))
    return  u0



comm = MPI.COMM_WORLD		# initialize MPI
p = comm.Get_rank()		# id of the current processor
nproc = comm.Get_size()		# number of processors

Lmin = 0 			# left domain of the whole problem
Lmax = 1.0			# right domain of the whole problem
Kappa = 0.1

npts = 16			# number of grid points in the current processor 
Ntotal = nproc*npts+1
dx = (Lmax-Lmin) / (Ntotal)

cfl = 0.4
dt = cfl * (dx**2/Kappa)
Nt = 10000



# root proc is responsible for initializing
if p == 0:
    x = np.linspace(Lmin,Lmax,num=Ntotal)
    u0 = initial_condition( x )
    # print('initial cond',u0)
else:
    x = None
    u0 = None

# last processor expects one extra element
if p == nproc-1:
    u_loc = np.zeros(npts+1)
else:
    u_loc = np.zeros(npts)

if p == 0:
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


def update_heat(p, nproc, dx, u_loc, Nt, dt, Kappa ):

    n = u_loc.shape[0]
    
    u = np.zeros(n+2)
    u[1:-1] = u_loc
    unew = u

    # main loop 
    for k in range(Nt):
      
        # message 1, send LEFT to RIGHT
        if (p < nproc-1):
            comm.send(u[n], dest = p+1, tag=1)


        if (0 < p):
            u[0] = comm.recv(source = p-1, tag=1)

   
        # message 2, send from RIGHT to LEFT
        if (0 < p):
            comm.send(u[1], dest=p-1, tag=2)


        if (p < nproc-1):
            u[n+1] = comm.recv(source=p+1, tag=2)

        # set Neumann BCs
        if (p == 0): u[0] = u[2]

        if (p==nproc-1): u[-1] = u[-3]


        unew[1:-1] = u[1:-1] + Kappa*dt/(dx**2)*( u[0:-2]-2*u[1:-1]+u[2:] )
        u = unew  

    return u

u_sol = update_heat(p, nproc, dx, u_loc, Nt, dt, Kappa)

print(u_sol)


# Root processor gather all data 
comm.Barrier()
if p == 0:
    print('Root processor to gather data')
    u_final = np.zeros(Ntotal)
else:
    u_final = None


comm.Gatherv(u_sol[1:-1],[u_final,sct,dis,MPI.DOUBLE])

if p==0:
    sio.savemat('heat_final.mat',{'u_final':u_final,'u0':u0,'x':x})



