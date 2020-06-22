#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
from numba import njit,prange
from timeit import timeit

# In[57]:


def ident_np_nojit(x):
    return np.cos(x) ** 2 + np.sin(x) ** 2

@njit
def ident_np(x):
    return np.cos(x) ** 2 + np.sin(x) ** 2

@njit
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


# In[51]:


A = np.arange(10000).reshape((100,100))
ident_np(A)
timeit.timeit('ident_np(A)', number = 10)


# In[52]:


ident_loops(A)
# get_ipython().run_line_magic('timeit', 'ident_loops(A)')


# In[53]:


ident_np2(A)
# get_ipython().run_line_magic('timeit', 'ident_np2(A)')


# In[54]:


ident_loops2(A)
# get_ipython().run_line_magic('timeit', 'ident_loops2(A)')


# In[55]:


# get_ipython().run_line_magic('timeit', 'ident_np_nojit(A)')


# In[ ]:




