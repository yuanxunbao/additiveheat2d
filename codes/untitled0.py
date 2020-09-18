#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:25:42 2020

@author: yigongqin
"""

from scipy.spatial.transform import Rotation as R
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
import statistics as stat
from scipy.stats import special_ortho_group
import time 

sfg = False
pfg = False
supch = False

#direct scan is for set that has itself

def apply_rot(coors):
    
    #quat = np.random.rand(d)
    #quat = quat/LA.norm( quat )
    #rot = R.from_quat(quat)
    #coors_new =  coors@(rot.as_matrix())

    rot = special_ortho_group.rvs(d)
    coors_new = np.dot(coors, rot)
    
    return coors_new

def sup_charg(k, old_list, center, Bi, coors):

    new_list = list(old_list)


    for i in range(k):

        pid = old_list[i]
        Vi = np.array(query( pid, Bi, coors )) # GIVE a 1d array that have k(L+1) coordinates
        
        if pfg==True: print('Vi length',len(Vi),'list',Vi)
        newpid = int(np.argwhere(Vi==pid))
        print('point id: ',pid,'; id in Vi: ',newpid)
        
        suspect_Vi, Dsuspi = direc_scan(len(Vi), k, 1, np.array([newpid]), coors[Vi,:])
        sus_suspect = Vi[suspect_Vi][0]  
        #print(sus_suspect,new_list)
        new_list = new_list + list(set(sus_suspect) - set(new_list))
    
    #new_list = old_list    
    
    len_new = len(new_list)
    
    if len_new<=k**2+k:
        print('supercharging size',len_new)
        subtra = np.repeat(np.reshape(coors[[ pid ],:],(1,d)),len_new,axis=0)
        norms = LA.norm( coors[new_list,:]- subtra ,axis = 1)
        neighbors = norms.argsort()[:k]
        Dtruei = LA.norm( norms[neighbors] )**2/k
        neighbor_list = np.array(new_list)[neighbors]         
        
    else: print('error occurs in supercharging!!!!!!!!!!!!!!!')
    
    
    
    return neighbor_list, Dtruei


def direc_scan(N, k, Nsam, samid, coors):
    
    
    neighbor_list = np.zeros((Nsam, k),dtype=int)
    Dtruei = np.zeros(Nsam) 
    
    for i in range(Nsam):
        
        print('point id:',samid[i])
        subtra = np.repeat(np.reshape(coors[[ samid[i] ],:],(1,d)),N,axis=0)
        norms = LA.norm( coors- subtra ,axis = 1)                   # should be a 1D array N 
        neighbors = norms.argsort()[1:k+1]
        Dtruei[i] = LA.norm( norms[neighbors] )**2/k
        neighbor_list[i,:] = neighbors
        
    Dtrue = np.mean(Dtruei)    
    return neighbor_list, Dtrue 


def improve(first_list, second_list ,coors, pid, Dsusp):
    
    
    new_list = first_list + list(set(second_list) - set(first_list))
    
    len_new = len(new_list)
    
    if len_new>k:
    
        subtra = np.repeat(np.reshape(coors[[ pid ],:],(1,d)),len_new,axis=0)
        norms = LA.norm( coors[new_list,:]- subtra ,axis = 1)
        neighbors = norms.argsort()[:k]
        Dtruei = LA.norm( norms[neighbors] )**2/k
        neighbor_list = np.array(new_list)[neighbors] 
        
    elif len_new ==k:
        neighbor_list = np.array(new_list)
        Dtruei = Dsusp
    else: print('something wrong in iterations!!!!!!!!!!')
    
    return neighbor_list, Dtruei


def RANN(N, k, Nsam, samid, coors, ite):
    
    old_list = np.zeros((Nsam, k),dtype=int)
    
    for Tii in range(ite):
        
        
        coors = apply_rot(coors)
        suspect_list = np.zeros((Nsam, k),dtype=int)
        Dsuspi = np.zeros(Nsam) 
        ini_list = np.arange(N)
        Bi=[]; Bi.append(ini_list)
        Bi = divide(1, Bi, coors)
        
        #return Bi    #check if get the right division
      
        for i in range(Nsam):
            #print(samid[i])
            pid = samid[i]
            Vi = np.array(query( pid, Bi, coors )) # GIVE a 1d array that have k(L+1) coordinates
            
            if pfg==True: print('Vi length',len(Vi),'list',Vi)
            newpid = int(np.argwhere(Vi==pid))
            print('point id: ',pid,'; id in Vi: ',newpid)
            
            suspect_Vi, Dsuspi[i] = direc_scan(len(Vi), k, 1, np.array([newpid]), coors[Vi,:])
            suspect_list[i,:] = Vi[suspect_Vi]
            # improve from oldList
            if Tii>1 : 
                suspect_list[i,:], Dsuspi[i] = improve(list(old_list[i,:]), list(suspect_list[i,:]), coors, pid, Dsuspi[i])
            
            old_list[i,:] = suspect_list[i,:]
    
        
    if supch == True: 
        
        for scid in range(Nsam):
            
                       
            old_list[i,:], Dsuspi[i] = sup_charg(k, old_list[scid,:], samid[scid], Bi, coors)
            
            
        
    Dsusp = np.mean(Dsuspi) 
    
    return old_list, Dsusp, Bi


def divide(ite, Bi, coors):   #
    
    Bnew = []

    for i in range(len(Bi)):
        #print(len(Bi[i]))
        minus = Bi[i][ coors[Bi[i],ite-1].argsort()[:int( len(Bi[i])/2 )] ]
        Bnew.append(minus)
        plus = np.setdiff1d(Bi[i],minus)
        Bnew.append( plus )
    if ite==L:
        #print(len(Bi))
        return Bnew
    
    else:
         return divide(ite+1, Bnew, coors)
    
def query( pid, Bi, coors ):
    
    gid = 0
    
    num_box = 2**L
    
    for i in range(num_box):
        
        if pid in Bi[i]: gid = i
        
        #if coors[pid,i] > stat.median(coors[:,i]):
            #gid += 2**(L -1 -i)
    
    
    # check if pid in gid
    if pid in Bi[gid]: pass
    else: print('not find the query point--------------------------!!!!!!')
   # else: 
        
    
    Vi= list(Bi[gid])
    #print(gid)
    #bingid = list(bin(gid)[2:].zfill(L))
    bingid = bin(gid)[2:].zfill(L)
    print('binary coordinates of box', bingid)
    
    neighborbox = list(bingid)
    for i in range(L):
        #neighborbox = bingid
        
        if neighborbox[i] =='1':
            
            neighborbox[i] = '0' 
            #print(neighborbox)
            newnb = int(''.join(neighborbox),2)
            Vi += list(Bi[newnb])
            neighborbox[i] = '1' 
        elif neighborbox[i] =='0':
            
            neighborbox[i] = '1'   
           # print(neighborbox)
            newnb = int(''.join(neighborbox),2)
            Vi += list(Bi[newnb])
            neighborbox[i] = '0' 
        else: print('error!!')
    #print(neighborbox)

  
    return  Vi              # list





# data sets N*d / translation/ rotation 


N = 122880#122880

Nsam = 100#100#100#3#2000   # need to generate Nsam * k suspects

samid = np.random.randint(N, size=Nsam )
#samid = np.array([7,53,33])

k = 15#15#7#15

L = int(np.log2(N/k))  #2

Titerat = 1;


d_arr = np.array([15,20,30,40,50,60,80,100,120,150,200])
#d_arr = np.array([15])
'''

N = 80#122880

Nsam = 3#100#100#3#2000   # need to generate Nsam * k suspects

samid = np.random.randint(N, size=Nsam )
#samid = np.array([7,53,33])

k = 7#15#7#15

L = 2#int(np.log2(N/k))  #2

Titerat = 2;


d_arr = np.array([2])

'''

numd = len(d_arr)
Ratio = np.zeros(numd)
prop = np.zeros(numd)


for dim in range(len(d_arr)):
    
    
    d = d_arr[dim] 
    #d = 2#10
    
    coors = np.random.multivariate_normal(np.zeros(d), np.identity(d), N)
    
    center = np.mean(coors, axis=0)
    
    
    # make center at the origin
    coors = coors - np.repeat(np.reshape(center,(1,d)),N,axis=0)
    print(np.mean(coors, axis=0))
    
    neighbor_list, Dtrue = direc_scan(N, k, Nsam, samid, coors)
    
    #Bi = RANN(N, k, Nsam, samid, coors)
    
    suspect_list, Dsusp, Bi  = RANN(N, k, Nsam, samid, coors, Titerat)
    
    
    Ratio[dim] = Dsusp/Dtrue
    diff = []
    for i in range(Nsam):
        
        diff += list(np.setdiff1d(neighbor_list[i],suspect_list[i]))
    
    prop[dim] = 1 - len(diff)/  (Nsam*k)
    
print('\nRatio = ', Ratio)
print('Proportion = ', prop)

    


'''
plt.scatter(coors[:,0],coors[:,1])
plt.xlim(-3,3);plt.ylim(-3,3)

if sfg == True: plt.savefig('all points',dpi=400)

fig1 = plt.figure(figsize=[12, 12])

ax1 = fig1.add_subplot(221)  
plt.scatter(coors[:,0],coors[:,1])
plt.scatter(coors[samid,0],coors[samid,1])
plt.xlim(-3,3);plt.ylim(-3,3)
#plt.title('point id'+str(samid[0])+'neighbor id'+str(neighbor_list[0,:]))
ax1 = fig1.add_subplot(222)  
plt.scatter(coors[samid[0],0],coors[samid[0],1])
plt.scatter(coors[neighbor_list[0,:],0],coors[neighbor_list[0,:],1])
plt.xlim(-3,3);plt.ylim(-3,3)
plt.title('point id'+str(samid[0])+'neighbor id'+str(neighbor_list[0,:]))
ax1 = fig1.add_subplot(223)  
plt.scatter(coors[samid[1],0],coors[samid[1],1])
plt.scatter(coors[neighbor_list[1,:],0],coors[neighbor_list[1,:],1])
plt.xlim(-3,3);plt.ylim(-3,3)
plt.title('point id'+str(samid[1])+'neighbor id'+str(neighbor_list[1,:]))
ax1 = fig1.add_subplot(224)  
plt.scatter(coors[samid[2],0],coors[samid[2],1])
plt.scatter(coors[neighbor_list[2,:],0],coors[neighbor_list[2,:],1])
plt.xlim(-3,3);plt.ylim(-3,3)
plt.title('point id '+str(samid[2])+' neighbor id  '+str(neighbor_list[2,:]))


if sfg == True: plt.savefig('direct scan',dpi=400) 

fig2 = plt.figure(figsize=[12, 12])

ax2 = fig2.add_subplot(221) 
plt.scatter(coors[Bi[1],0],coors[Bi[1],1])
plt.xlim(-3,3);plt.ylim(-3,3)
ax2 = fig2.add_subplot(222) 
plt.scatter(coors[Bi[3],0],coors[Bi[3],1])
plt.xlim(-3,3);plt.ylim(-3,3)
ax2 = fig2.add_subplot(223) 
plt.scatter(coors[Bi[0],0],coors[Bi[0],1])
plt.xlim(-3,3);plt.ylim(-3,3)
ax2 = fig2.add_subplot(224) 
plt.scatter(coors[Bi[2],0],coors[Bi[2],1])
plt.xlim(-3,3);plt.ylim(-3,3)


if sfg == True: plt.savefig('division',dpi=400)

fig3 = plt.figure(figsize=[12, 12])
ax1 = fig3.add_subplot(221)  
plt.scatter(coors[:,0],coors[:,1])
plt.scatter(coors[samid,0],coors[samid,1])
plt.xlim(-3,3);plt.ylim(-3,3)
#plt.title('point id'+str(samid[0])+'neighbor id'+str(suspect_list[0,:]))
ax1 = fig3.add_subplot(222)  
plt.scatter(coors[samid[0],0],coors[samid[0],1])
plt.scatter(coors[suspect_list[0,:],0],coors[suspect_list[0,:],1])
plt.xlim(-3,3);plt.ylim(-3,3)
plt.title('point id'+str(samid[0])+'neighbor id'+str(suspect_list[0,:]))
ax1 = fig3.add_subplot(223)  
plt.scatter(coors[samid[1],0],coors[samid[1],1])
plt.scatter(coors[suspect_list[1,:],0],coors[suspect_list[1,:],1])
plt.xlim(-3,3);plt.ylim(-3,3)
plt.title('point id'+str(samid[1])+'neighbor id'+str(suspect_list[1,:]))
ax1 = fig3.add_subplot(224)  
plt.scatter(coors[samid[2],0],coors[samid[2],1])
plt.scatter(coors[suspect_list[2,:],0],coors[suspect_list[2,:],1])
plt.xlim(-3,3);plt.ylim(-3,3)
plt.title('point id '+str(samid[2])+' neighbor id  '+str(suspect_list[2,:]))



if sfg == True:  plt.savefig('RANN',dpi=400)

'''




# loop  over d

