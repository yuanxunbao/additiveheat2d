import os
#must set these before loading numpy:
os.environ["OMP_NUM_THREADS"] = '4' # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = '4' # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = '4' # export MKL_NUM_THREADS=6
#os.environ["VECLIB_MAXIMUM_THREADS"] = '4' # export VECLIB_MAXIMUM_THREADS=4
#os.environ["NUMEXPR_NUM_THREADS"] = '4' # export NUMEXPR_NUM_THREADS=6


os.environ["NUMBA_NUM_THREADS"] = '1'

import numpy as np
import time
from numba import njit

# from grad_routine_nojit import * 

def gradxx(h_in,v):

	return  h_in*( v[1:-1,1:] - v[1:-1,:-1] )

@njit()
def gradxx_jit(h_in,v):

	return  h_in*( v[1:-1,1:] - v[1:-1,:-1] )

@njit(parallel=True)
def gradxx_par(h_in,v):

	return  h_in*( v[1:-1,1:] -  v[1:-1,:-1] )


nx = 160
ny = 640
h = 1.0
Mt = 3003

phi = np.random.randn(nx,ny)

np.show_config()



# numpy
start = time.time()
for i in range(Mt):

	gradxx(h,phi)

end = time.time()

print('time numpy = ', (end-start))

# numba
gradxx_jit(h,phi)
start = time.time()
for i in range(Mt):

	gradxx_jit(h,phi)

end = time.time()

print('time numba = ', (end-start))


# numba parallel
gradxx_par(h,phi)
start = time.time()
for i in range(Mt):

	gradxx_par(h,phi)

end = time.time()

print('time numba par = ', (end-start))

