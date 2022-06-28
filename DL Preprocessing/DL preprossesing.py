#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 02:44:56 2021

@author: mohammadkazemzadeh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import time
from sklearn.preprocessing import normalize


aa=keras.models.load_model('./epochFTO4(LogCosh-2s)-93')
print('finish loading the DL model!!!')
name=['LPS-P.aeruginosa.csv','Pyocynin.csv','PGN-SA.csv','LTA-SA.csv','LPS-KPS.csv']

nam='LPS-P.aeruginosa.csv'

for nam in name:
        
    a=pd.read_csv(nam,header=None)
    a=np.array(a)
    
    
    y=[]
    yt=[]
    for i in range(len(a)):        
        y=np.interp(range(1000),np.linspace(0,999,len(a[0])),a[i])
        yt.append(y)
        if i%100==0:
            print(i/len(a)*100)
            
    yt=np.array(yt)
    
    
    X=yt
    X=np.array(X)
    X=X.reshape(len(X),len(X[0]),1)
    s=time.time()
    Y=aa.predict(X)
    e=time.time()
    print(e-s)
    Y[0]=normalize(Y[0])
    np.savetxt(nam[:-4]+'-pre.csv',Y[0],delimiter=',')
    plt.plot(Y[0][0])
    plt.show()
    print("finish analyzing %s!!!"%nam)
    
    
