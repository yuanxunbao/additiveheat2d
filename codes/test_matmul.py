import os

os.environ["MKL_NUM_THREADS"] = '56' # export MKL_NUM_THREADS=6

import numpy as np  
from numba import njit
import time   
N = 1600 
M = 6400 
  
# k_list = [64, 80, 96, 104, 112, 120, 128, 144, 160, 176, 192, 200, 208, 224, 240, 256, 384]  

K = 1600
  
def get_gflops(M, N, K):  
    return M*N*(2.0*K-1.0) / 1000**3  
    # return M*N/1000**3	  


@njit()
def numba_matmul(A,B):
    
    return A*B


np.show_config()  
  

a = np.array(np.random.random((M, N)), dtype=np.single, order='C', copy=False)  
b = np.array(np.random.random((N, K)), dtype=np.single, order='C', copy=False)  
A = np.matrix(a, dtype=np.single, copy=False)  
B = np.matrix(b, dtype=np.single, copy=False)  

# 1st run for numba to compile  
C = np.matmul(A,B) 
  
start = time.time()  
for i in range(5):
    C = np.matmul(A,B) 

end = time.time()  
  
tm = (end-start) / 5.0  
  
print ('{0:9.7} sec, {1:9.7} gflops/sec'.format(tm, get_gflops(M, N, K) / tm))
    

start2 = time.time()

for i in range(5):
    C = A*B
    
end2 = time.time()

tm2 = (end2-start2) / 5.0

print ('{0:9.7} sec, {1:9.7} gflops/sec'.format(tm2, get_gflops(M, N, K) / tm2))
                        
    
