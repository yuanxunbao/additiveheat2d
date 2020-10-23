#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 00:15:20 2020

@author: yuanxun
"""


import os


Q  = [1250, 1500, 1750, 2000] # Q
rb = [0.2e-3, 0.4e-3, 0.6e-3, 0.8e-3] 
Vs = [0.2e-3, 0.4e-3, 0.6e-3, 0.8e-3]



for p1 in Q:
    for p2 in rb:
        for p3 in Vs:
            print('[Q,rb,Vs] = [%d,%.2e,%.2e]'%(p1,p2,p3) )
            os.system('python3 macro_heat_spot_weld.py %d %.3e %.3e'%(p1,p2,p3))