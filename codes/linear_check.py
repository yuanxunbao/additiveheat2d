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

dataname = 'heat2d_lx0.024_nx213_asp0.5_dt1.00e-02_Mt650.mat'
trajname = 'traj_low_try.mat'

t0 = 3.2

G_arr = load(dataname)['G_arr']
R_arr = load(dataname)['R_arr']
temp = load(dataname)['temp213']
xx = load(dataname)['xx']
yy = load(dataname)['yy']

nx = int(load(dataname)['nx'])
ny = int(load(dataname)['ny'])

lx = xx[0,-1]
ly = -yy[-1,0]


x = np.linspace(0,lx,nx)
ycoor = np.linspace(0,-ly,ny)

targ = 34  # select a point 

xtarg = load(trajname)['X_']
ytarg = load(trajname)['Y_']
G_read = load(trajname)['G_']
R_read = load(trajname)['R_']
time_read = load(trajname)['time_']
t_st = load(trajname)['t_st']

Gtarg = G_read[targ,:]   #high speed 8
Rtarg = R_read[targ,:]
xtarg = xtarg[targ,:]
ytarg = ytarg[targ,:]
T_len = len(xtarg)
Ttarg = np.zeros(T_len)
T_FR = np.zeros(T_len)
zl = np.zeros(T_len)

T_aa = np.zeros((T_len,T_len))
T_fra = np.zeros((T_len,T_len))

t_arr = time_read[0,:]

dt = t_arr[1] - t_arr[0]

for i in range(T_len):
    
    zl[i] = np.sqrt( (xtarg[i]-xtarg[0])**2 + (ytarg[i]-ytarg[0])**2 )


def tem_dis(tp):
    time = int(t0/dt)+ t_st + tp +1  #479, 349
    
    G = np.reshape(G_arr[:,time],(ny,nx),order='F')
    T = np.reshape(temp[:,time],(ny,nx),order='F')
    
    Titp = itp2d( x, ycoor, T, kind = 'cubic' )
    Gitp = itp2d( x, ycoor, G, kind = 'cubic' )
    
    
    
    for i in range(T_len):
        
        Ttarg[i] = Titp(xtarg[i],ytarg[i])
        
    
    G0 = Gitp(xtarg[tp],ytarg[tp])
    
    
    print(tp,G0)
    
    for i in range(T_len):
        

        
        T_FR[i] = Ttarg[tp] + G0*(zl[i]-zl[tp])
        

    return Ttarg, T_FR


for nt in range(T_len):
    
    T_aa[nt,:] , T_fra[nt,:] = tem_dis(nt)
    
    
    



fig3 = plt.figure(figsize=[15,3])



ax4 = fig3.add_subplot(131)
ax4.plot(zl,T_aa[0,:])
ax4.plot(zl,T_fra[0,:])
plt.xlabel('z');plt.ylabel('T');plt.title('t=0')
plt.legend(('actual','FR'))



ax5 = fig3.add_subplot(132)
ax5.plot(zl,T_aa[5,:])
ax5.plot(zl,T_fra[5,:])
plt.xlabel('z');plt.ylabel('T');plt.title('t=5e-5')
plt.legend(('actual','FR'))


ax6 = fig3.add_subplot(133)
ax6.plot(zl,T_aa[17,:])
ax6.plot(zl,T_fra[17,:])
plt.xlabel('z');plt.ylabel('T');plt.title('t=10e-5')
plt.legend(('actual','FR'))



#save('FR_check_low.mat',{'r':zl,'t_ma':t_arr,'T_actual':T_aa,'T_FR':T_fra,'G_t':Gtarg,'R_t':Rtarg})






