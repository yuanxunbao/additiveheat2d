#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:45:21 2020

@author: yigongqin
"""

# macro model/ with latent heat
# crank-nicolson/ center difference
# linear solver CG

import os
import numpy as np
from macro_param_low2 import phys_parameter, simu_parameter 
from scipy import sparse as sp
from scipy.sparse import linalg as la
import matplotlib.pyplot as plt
from scipy.io import savemat as save
from math import pi
#from sksparse.cholmod import cholesky
import time
from scipy.optimize import fsolve
from scipy.interpolate import interp2d as itp2d



#from scipy.sparse import csc_matrix
class gmres_counter(object):
    def __init__(self, TOL):
       # self._disp = True
        self.niter = 0
        self.tol = TOL
    def __call__(self, rk=None):
        self.niter += 1
        if rk<self.tol:
            print('gmres iteration %3i\trk = %s' % (self.niter, str(rk)))
            
def heat_source(t,dx):
    #print('nx=',nx,'amazing!')
    
    #return p.q0 * np.exp( -2*( np.arange(nx).reshape((nx,1))*dx - l0 - p.Vs*t )**2/p.rb**2 )
    q0_decay = energy_decay(t, t0)
    #return p.q0 * np.exp( -2*( np.arange(nx).reshape((nx,1))*dx - l0 )**2/p.rb**2 )
    return q0_decay * np.exp( -2*( np.arange(nx).reshape((nx,1))*dx - l0 )**2/p.rb**2 )

def Up_boundary(t,dx,Tu):
    
    qs = heat_source(t,dx)
    
    return -2*dx/p.K*( -qs + p.hc*(Tu-p.Te) + p.epsilon*p.sigma*(Tu**4-p.Te**4) )

def imex_latent(y):
    
    down = y>p.Ts
    up = y<p.Tl
    
    y_l = y*down*up
        
    
    ind = np.ones((nv,1))*down*up
    ind = np.reshape(ind,(nv))
    
    
    # a more smooth form of the fluid fraction function
    factor = pi/2*np.sin(pi*(y-p.Ts)/(p.Tl-p.Ts))
    y_l = y_l*factor
    ind = ind*np.reshape(factor,(nv))
    
    A_l = sp.diags(ind,format='csc')
    

    
    
    return y_l, A_l




def gradients(y_new,y_old,h):
    # yn is new result, yp is previous time step
    Tn = np.reshape(y_new,(ny,nx),order='F')
    T_old = np.reshape(y_old,(ny,nx),order='F')
    
    #gradT_n = np.ones((ny,nx))
    gradT_x = np.ones((ny,nx)); gradT_y = np.ones((ny,nx)); 
    gradT_x[1:-1,1:-1] = ( Tn[1:-1,2:] - Tn[1:-1,:-2])/(2*h)
    gradT_y[1:-1,1:-1] = ( -Tn[2:,1:-1] + Tn[:-2,1:-1])/(2*h)
    gradT_n = np.sqrt( gradT_x**2 + gradT_y**2 )
    
    # set boundary condition for G
    
    #gradT_n[0,:] = 
    #gradT_n[-1,:] = gradT_n[-1,:]
    
    
    #grady_n = np.reshape(gradT_n,(nv,1),order='F') 
    
    #dTdt = ( y_new - y_old )/dt
    dTdt = ( Tn - T_old )/dt
    
    return gradT_n, - dTdt/gradT_n, gradT_x, gradT_y



def liquid_contour(Temp): # this is only for selecting initial points
    
    T = np.reshape(Temp,(ny,nx),order='F') 
    
    
    xj_arr = []
    yj_arr = []
    Tj_arr = []
    
    

    
    for ii in range(nx):
        jj = 0
        while(T[jj,ii]>=p.Tcon): jj +=1  
        if yy[jj,0]< -20e-6:
            #xj_arr = np.concatenate((xj_arr, xx[0,ii]))
            #yj_arr = np.concatenate((yj_arr, yy[jj,0]))
            #Tj_arr = np.concatenate((Tj_arr, T[jj,ii]))
            xj_arr.append(xx[0,ii])
            yj_arr.append(yy[jj,0])
            Tj_arr.append(T[jj,ii])
            #Tj_arr[ii] = T[jj,ii]   
        #Gj_arr[ii] = G[jj,ii]
        #Rj_arr[ii] = R[jj,ii]
        
    # compute thetaj_arr
    #thetaj_arr[0] = (yj_arr[1]-yj_arr[0])/dx
    #thetaj_arr[-1] = (yj_arr[-1]-yj_arr[-2])/dx
    #thetaj_arr[1:-1] = (yj_arr[2:]-yj_arr[:-2])/(2*dx)
    #thetaj_arr = np.arctan(thetaj_arr) 
    #Radj_arr = np.sqrt( (l0-xj_arr)**2+(yj_arr)**2 )    
    #thetaj_arr = np.arctan((l0-xj_arr)/yj_arr)*180/pi
    num_sam = len(xj_arr)
    
    Gj_arr = np.zeros(num_sam)
    Rj_arr = np.zeros(num_sam)
    betaj_arr = np.zeros(num_sam)
    
    return np.array(xj_arr), np.array(yj_arr), np.array(Tj_arr), Gj_arr, Rj_arr, betaj_arr


def QoI_save(t,xj_arr,yj_arr,Tj_arr,Gj_arr,Rj_arr, betaj_arr):
    

    # open one file and write the time, coordinates and G, R, theta
            
    depth = yj_arr[int((len(yj_arr)-1)/2)]    
    
    with open(s.outname,'w' if t==0 else 'a') as out_file:
        
        if t>t0 and depth < -40e-6:
            
            for iii in range(len(yj_arr)):
    
                    
                out_string = ''
                out_string += str('%5.3e'%t)
                out_string += ',' + str('%5.3e'%xj_arr[iii])
                out_string += ',' + str('%5.3e'%yj_arr[iii])
                out_string += ',' + str('%5.3e'%Tj_arr[iii])
                out_string += ',' + str('%5.3e'%Gj_arr[iii])
                out_string += ',' + str('%5.3e'%Rj_arr[iii])
                #out_string += ',' + str('%5.3e'%Radj_arr[iii])
                out_string += ',' + str('%5.3e'%betaj_arr[iii])
                out_string += '\n'
                out_file.write(out_string)
        
    
    
    return 



def energy_decay(t, t0 ):
    
    # energy start to decay at the time t0 with the time scale parameter tau_0: rb/vs
    if t< t0 : q0_decay = p.q0
    else: q0_decay = p.q0*np.exp(-2*(p.Vs*( t - t0 ))**2/p.rb**2)
            
        
    return   q0_decay



# calculate length of step
def stepj(ss, *arg):  
    # here gT_xn and gT_yn are interpolated and T_itp is interpolated temperature at next time step
    # here xj_n and yj_n are arrays
    xj_n, yj_n, T_itp, gT_xn, gT_yn = arg
    
    xj_np = xj_n + ss*gT_xn
    yj_np = yj_n + ss*gT_yn
    
    
    Tj_np = T_itp(xj_np, yj_np)
    
    
    return Tj_np - p.Tcon



def trajectory(xj_arrn, yj_arrn, gTx_n, gTy_n, R, y_n, y_np):  
    # calculate coordinates for time at n+1 and output interpered gradTx gradTy, R 
    
    T_n = np.reshape(y_n,(ny,nx),order='F') 
    T_np = np.reshape(y_np,(ny,nx),order='F') 
    
    num_sam = len(xj_arrn)
    
    
    Tj_arrn = np.zeros(num_sam)
    gTxj_arrn = np.zeros(num_sam)
    gTyj_arrn = np.zeros(num_sam)
    Rj_arrn = np.zeros(num_sam)
    sj_arr = np.zeros(num_sam)
    
    T_itp = itp2d( x, ycoor, T_np, kind = 'cubic' ) # get interp object for next time step t_n+1

    
    gTxn_itp = itp2d( x, ycoor, gTx_n, kind = 'cubic' ) 
    gTyn_itp = itp2d( x, ycoor, gTy_n, kind = 'cubic' ) 
    R_itp = itp2d( x, ycoor, R, kind = 'cubic' ) 
    Tn_itp = itp2d( x, ycoor, T_n, kind = 'cubic' )
    
    for ii in range(num_sam):
        
        x_j = xj_arrn[ii];
        y_j = yj_arrn[ii]
        Tj_arrn[ii] = Tn_itp(x_j,y_j)
        gTxj_arrn[ii] = gTxn_itp(x_j,y_j)
        gTyj_arrn[ii] = gTyn_itp(x_j,y_j)
        Rj_arrn[ii] = R_itp(x_j,y_j)

        arg = ( x_j, y_j, T_itp, gTxj_arrn[ii], gTyj_arrn[ii] )
        sj_arr[ii] = fsolve( stepj, [0.0], args=arg)
    
    Gj_arrn = np.sqrt( gTxj_arrn**2 + gTyj_arrn**2 )
    betaj_arrn = np.arctan( gTyj_arrn/gTxj_arrn)*180/pi
    
    return  xj_arrn + sj_arr*gTxj_arrn, yj_arrn + sj_arr*gTyj_arrn,\
        xj_arrn, yj_arrn, Tj_arrn, Gj_arrn, betaj_arrn, Rj_arrn





start = time.time()

#====================parameters==========================

p = phys_parameter()
s = simu_parameter(p.rb)

lx = s.lxd
aratio = s.asp_ratio
ly = lx*aratio


## make xx, yy, the dimensions and index 

nx = s.nx
dx = lx/(nx-1)
print('mesh width is', dx)

ny = int((nx-1)*aratio+1)
print('y grids ny:',ny)
#nx= N+1  #x grids;ny= M+1  #y grids
nv= ny*nx #number of variables

x = np.linspace(0,lx,nx)
ycoor = np.linspace(0,-ly,ny)

xx,yy = np.meshgrid(x,ycoor)



dt = s.dt
C = p.alpha*dt/dx**2  #CFL number

lat = p.Lm/(p.Tl-p.Ts)  # non dimensionalized latent heat

l0 = lx/2 #laser starting point 


Mt= s.Mt
Tt = Mt*dt
t=0


# macro output sampling arrays



# set the power source to decay
Tt_arr = np.linspace(0, Tt-dt, Mt)

q0_arr = np.zeros(Mt)

t0 = s.t0

tstop = Tt


#tau_0 = 0.1*t0

for ii in range(Mt):
    q0_arr[ii] = energy_decay(Tt_arr[ii], t0)
 
fig5 = plt.figure()
plt.plot(Tt_arr, q0_arr )


# ===================Sampling parameters=================
nts = s.nts
kts = int( Mt/nts )
#=====================Generate matrix========================

I = sp.eye(nx*ny,format='csc')
Ix = sp.eye(nx,format='csc'); Iy = sp.eye(ny,format='csc')
Ix[0,0] = 0.5; Ix[-1,-1] = 0.5
Iy[0,0] = 0.5; Iy[-1,-1] = 0.5


Dxx = sp.diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx)).toarray()
Dxx[0,1]=2; Dxx[-1,-2]=2 #Riemann boundary condition 

Dyy = sp.diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny)).toarray()
Dyy[0,1]=2; Dyy[-1,-2]=2


L = sp.kronsum(Dyy,Dxx,format='csc')

Q =sp.kron(Ix,Iy,format='csc')

A0 = Q@(I - C/2.0*L)
#Aarr = A.toarray()

#pd =is_pos_def(A.toarray())   #check if A matrix is positive definite 
# LU decomposition Pr*A*Pc = LU
#lu = la.splu(A0)
#factor = cholesky(A0)

#LU_x = lambda x: factor(x)
#pvec = la.LinearOperator((nv, nv), LU_x)


##====================Initial condition=========================
Temp = np.zeros((nv,nts+1))
G_arr = np.zeros((nv,nts+1)); R_arr = np.zeros((nv,nts+1))
#T = np.zeros((ny,nx))      # temperature T0
#T = np.arange(nv).reshape((ny,nx))
T0 = p.Te*np.ones((ny,nx))
y = np.reshape(T0,(nv,1),order='F')  #T0/y0
Temp[:,[0]] = y

#Gv,Rv = gradients(y,y,dx)

#G_arr[:,[0]] = Gv; R_arr[:,[0]] = Rv

y_l,A_l=imex_latent(y)
#plt.imshow(T0,cmap=plt.get_cmap('hot'))

Tu = y[0:-1:ny]
U = Up_boundary(t,dx,Tu)          # U0

bU = np.zeros((nv,1))
bU[0:-1:ny] = U

#======================time evolusion=======================

TOL = 1e-9 #gmres tolerance


for ii in range(Mt):
    
    # obtain right hand side
    b = y + lat*y_l + C*bU + C/2.0*L@y           # b_n-1 from T_n-1 and U_n-1
    
    
    A = A0 + lat*Q@A_l
    #A_x = lambda x: A@x
    #mvec = la.LinearOperator((nv, nv), A_x)
    
    #counter = gmres_counter(TOL)
    #ynew,indix = la.gmres(A,Q@b,x0=y,tol=TOL,M=None,callback=counter)        # T_n
    ynew,indix = la.cg( A, Q@b, x0=y, tol=TOL, M=None)#callback=counter)
    ynew=np.reshape(ynew,((nv,1)))
    
    G,R, gTx, gTy = gradients(ynew, y, dx)
    
    #t += dt
 
    if np.absolute(t-s.ts)<1e-10:
        xj_arr, yj_arr, Tj_arr, Gj_arr, Rj_arr, betaj_arr = liquid_contour(ynew) 
    
    if t>s.ts and t<tstop:  # save QoIs
        xj_arr, yj_arr, xj_old, yj_old, Tj_arr, Gj_arr, betaj_arr, Rj_arr \
            = trajectory(xj_arr, yj_arr, gTx, gTy, R, y, ynew)
        QoI_save(t,xj_old,yj_old,Tj_arr,Gj_arr,Rj_arr, betaj_arr)

    
    y = ynew
    # latent heat
    y_l,A_l=imex_latent(y)    
   # Up boundary
    Tu = y[0:-1:ny]
    U = Up_boundary(t,dx,Tu)   # U_n
    #plug in right hand
    bU[0:-1:ny] = U
    
    t += dt

    
    if (ii+1)%kts==0:
       kk = int(np.floor((ii+1)/kts))
       print('=================================') 
       print('now time is ',t) 
       print('=================================') 
       Temp[:,[kk]] = y
       G_arr[:,[kk]] = np.reshape(G,(nv,1),order='F') 
       R_arr[:,[kk]] = np.reshape(R,(nv,1),order='F') 
       #print('gmres iterations: ',counter)
Tf = np.reshape(y,(ny,nx),order='F')
print(Tf[0,int(nx/4)],Tf[0,int(nx/2)])





fig2 = plt.figure(figsize=[12,4])
ax2 = fig2.add_subplot(121)
plt.imshow(Tf,cmap=plt.get_cmap('hot'))
plt.colorbar()

ax3 = fig2.add_subplot(122)
ax3.plot(xj_arr,yj_arr)
ax3.set_aspect('equal')
ax3.set_ylim(-ly, 0)
ax3.set_xlim(0, lx)

fig3 = plt.figure(figsize=[12,4])
ax4 = fig3.add_subplot(121)
plt.imshow(G[1:-1,1:-1],cmap=plt.get_cmap('hot'))
plt.xlim()
plt.colorbar();plt.title('G')

ax5 = fig3.add_subplot(122)
ax5.plot(xj_arr,yj_arr)
plt.imshow(R[1:-1,1:-1],cmap=plt.get_cmap('hot'))
plt.colorbar();plt.title('R')




#filename = 'Qlatdt' + str(dt)+'xgrids'+str(nx)+'.mat'
tempname = 'temp'+str(nx)

#save(os.path.join(s.direc,s.filename),{tempname: Temp,'G_arr':G_arr,'R_arr':R_arr,'nx':nx,'ny':ny,'xx':xx,'yy':yy})


end =time.time()
print('time used: ',end-start)







