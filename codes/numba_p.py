#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
from numba import njit,prange
import time


def ident_np_nojit(x):
    return np.cos(x) ** 2 + np.sin(x) ** 2

@njit()
def ident_np(x):
    return np.cos(x) ** 2 + np.sin(x) ** 2

@njit()
def ident_loops(x):
    r = np.empty_like(x)
    n = len(x)
    for i in range(n):
        r[i] = np.cos(x[i]) ** 2 + np.sin(x[i]) ** 2
    return r

@njit(parallel=True)
def ident_np2(x):
    return np.cos(x) ** 2 + np.sin(x) ** 2


@njit(parallel=True)
def ident_loops2(x):
    r = np.empty_like(x)
    n = len(x)
    for i in prange(n):
        r[i] = np.cos(x[i]) ** 2 + np.sin(x[i]) ** 2
    return r


nloop = 100
nsize = 1000
A = np.arange(nsize*nsize).reshape((nsize,nsize))


start=time.time()
for i in range(nloop):
    ident_np_nojit(A)
end=time.time()
print('numpy,native = ' ,(end - start)/nloop)

ident_np(A)
start=time.time()
for i in range(nloop):
    ident_np(A)
end=time.time()
print('numpy,jit = ' ,(end - start)/nloop)



ident_loops(A)
start=time.time()
for i in range(nloop):
    ident_loops(A)
end=time.time()
print('python,jit = ' ,(end - start)/nloop)



ident_np2(A)
start=time.time()
for i in range(nloop):
    ident_np2(A)
end=time.time()
print('numpy,jit,parallel = ' ,(end - start)/nloop)



ident_loops2(A)
start=time.time()
for i in range(nloop):
    ident_loops2(A)
end=time.time()
print('python,jit,parfor = ' ,(end - start)/nloop)




