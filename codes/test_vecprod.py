import os

os.environ["MKL_NUM_THREADS"] = '56' # export MKL_NUM_THREADS=6

import numpy as np  
import time   
N = 1600 
M = 6400
  
def get_gflops(M, N):  
    #return M*N*(2.0*K-1.0) / 1000**3  
    return M*N/1000**3	  

np.show_config()  
  
a = np.array(np.random.random((M*N)), dtype=np.single, order='C', copy=False)  
b = np.array(np.random.random((M*N)), dtype=np.single, order='C', copy=False)  
#A = np.matrix(a, dtype=np.single, copy=False)  
#B = np.matrix(b, dtype=np.single, copy=False)  
  
    
start = time.time()  

# multiply two vecs
for i in range(5):
    C = a * b 
  
end = time.time()  
  
tm = (end-start) / 5.0  
  
print ('{0:9.7} sec, {1:9.7} gflops/sec'.format(tm, get_gflops(M, N) / tm))
                        
 
