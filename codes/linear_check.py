#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 17:01:50 2020

@author: yigongqin
"""


import os
import numpy as np
from macro_param import phys_parameter, simu_parameter 
from scipy import sparse as sp
from scipy.sparse import linalg as la
import matplotlib.pyplot as plt
from scipy.io import savemat as save
from scipy.io import loadmat as load
from math import pi
#from sksparse.cholmod import cholesky

from scipy.optimize import fsolve
from scipy.interpolate import interp2d as itp2d


G_arr = load('heat2d_lx0.0008_nx213_asp0.5_dt1.00e-05_Mt400.mat')['G_arr']
R_arr = load('heat2d_lx0.0008_nx213_asp0.5_dt1.00e-05_Mt400.mat')['R_arr']
temp = load('heat2d_lx0.0008_nx213_asp0.5_dt1.00e-05_Mt400.mat')['temp213']
xx = load('heat2d_lx0.0008_nx213_asp0.5_dt1.00e-05_Mt400.mat')['xx']
yy = load('heat2d_lx0.0008_nx213_asp0.5_dt1.00e-05_Mt400.mat')['yy']

nx = int(load('heat2d_lx0.0008_nx213_asp0.5_dt1.00e-05_Mt400.mat')['nx'])
ny = int(load('heat2d_lx0.0008_nx213_asp0.5_dt1.00e-05_Mt400.mat')['ny'])

lx = 800e-6
ly = 400e-6

x = np.linspace(0,lx,nx)
ycoor = np.linspace(0,-ly,ny)



xtarg = load('trajectory.mat')['X_']
ytarg = load('trajectory.mat')['Y_']
targ = 34
xtarg = xtarg[targ,8:]
ytarg = ytarg[targ,8:]
T_len = len(xtarg)
Ttarg = np.zeros(T_len)
T_FR = np.zeros(T_len)
zl = np.zeros(T_len)



def tem_dis(tp):
    time = 348 + tp
    
    G = np.reshape(G_arr[:,time],(ny,nx),order='F')
    T = np.reshape(temp[:,time],(ny,nx),order='F')
    
    Titp = itp2d( x, ycoor, T, kind = 'cubic' )
    Gitp = itp2d( x, ycoor, G, kind = 'cubic' )
    
    
    
    for i in range(T_len):
        
        Ttarg[i] = Titp(xtarg[i],ytarg[i])
        
    
    G0 = Gitp(xtarg[tp],ytarg[tp])
    
    
    
    
    for i in range(T_len):
        
        if i>tp: zl[i] = np.sqrt( (xtarg[i]-xtarg[tp])**2 + (ytarg[i]-ytarg[tp])**2 )
        else: zl[i] = -np.sqrt( (xtarg[i]-xtarg[tp])**2 + (ytarg[i]-ytarg[tp])**2 )
        
        T_FR[i] = Ttarg[tp] + G0*zl[i]

    return Ttarg, T_FR, zl



fig3 = plt.figure(figsize=[15,3])

Ttarg, T_FR, zl = tem_dis(0)


ax4 = fig3.add_subplot(131)
ax4.plot(zl,Ttarg)
ax4.plot(zl,T_FR)
plt.xlabel('z');plt.ylabel('T');plt.title('t=0')
plt.legend(('actual','FR'))

Ttarg, T_FR, zl = tem_dis(5)

ax5 = fig3.add_subplot(132)
ax5.plot(zl,Ttarg)
ax5.plot(zl,T_FR)
plt.xlabel('z');plt.ylabel('T');plt.title('t=5e-5')
plt.legend(('actual','FR'))

Ttarg, T_FR, zl = tem_dis(10)

ax6 = fig3.add_subplot(133)
ax6.plot(zl,Ttarg)
ax6.plot(zl,T_FR)
plt.xlabel('z');plt.ylabel('T');plt.title('t=10e-5')
plt.legend(('actual','FR'))










