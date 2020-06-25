import os
#must set these before loading numpy:
os.environ["OMP_NUM_THREADS"] = '8' # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = '8' # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = '1' # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = '4' # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = '16' # export NUMEXPR_NUM_THREADS=6

import numpy as np
import numexpr as ne
import time

def get_gflops(M, N):
    return M*N / 1000**3

#np.__config__.show() #looks like I have MKL and blas
# np.show_config()

start_time=time.time()
#test script:


print('--------')
n = 1000
n2 = n**2
a = np.random.randn(n2).reshape(n,n)
b = np.random.randn(n2).reshape(n,n)
c = np.random.randn(n, 1)
ran_time=time.time()-start_time
#print("time to complete random matrix generation was %s seconds" % ran_time)

#np.dot(a,b)



start = time.time()
np.dot(a,b) 
end = time.time()


tm = end - start

print('\ntime = ', tm )
#print('flops = ' , get_gflops(n,n) / tm)

