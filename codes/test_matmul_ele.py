import os

os.environ["MKL_NUM_THREADS"] = '56' # export MKL_NUM_THREADS=6

import numpy as np  
import time   
N = 1600 
M = 6400
  
# k_list = [64, 80, 96, 104, 112, 120, 128, 144, 160, 176, 192, 200, 208, 224, 240, 256, 384]  

k_list = [160]
  
def get_gflops(M, N, K):  
    #return M*N*(2.0*K-1.0) / 1000**3  
    return M*N/1000**3	  

np.show_config()  
  
for K in k_list:  
    a = np.array(np.random.random((M, N)), dtype=np.single, order='C', copy=False)  
    b = np.array(np.random.random((M, N)), dtype=np.single, order='C', copy=False)  
    A = np.matrix(a, dtype=np.single, copy=False)  
    B = np.matrix(b, dtype=np.single, copy=False)  
  
    C = np.multiply(a,b) 
  
    start = time.time()  
  
    C = np.multiply(a,b)  
    C = np.multiply(a,b) 
    C = np.multiply(a,b)
    C = np.multiply(a,b)
    C = np.multiply(a,b)
  
    end = time.time()  
  
    tm = (end-start) / 5.0  
  
    print ('{0:4}, {1:9.7}, {2:9.7}'.format(K, tm, get_gflops(M, N, K) / tm))
                        
 
