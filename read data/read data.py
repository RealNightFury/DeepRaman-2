#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 23:32:12 2022

@author: elhamebr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


name=['LPS-P.aeruginosa.csv','Pyocynin.csv','PGN-SA.csv','LTA-SA.csv','LPS-KPS.csv']


for nam in name:
        
    a=pd.read_csv(nam,sep='\t')
    data=[]
    x=a[a.columns[0]]
    for i in range(len(a.columns)-1):
        data.append(a[a.columns[i+1]])
    
    a=np.array(data)
    np.savetxt(nam[:-4]+'-stn.csv',a,delimiter=',')