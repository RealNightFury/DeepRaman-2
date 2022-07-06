#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 14:01:03 2022

@author: elhamebr
"""



import pandas as pd
import numpy as np


name_of_excel_files=['Pyocynin.xlsx','PGN-SA.xlsx','LTA-SA.xlsx','LPS-KPS.xlsx','LPS-P.aeruginosa.xlsx']
for nam in name_of_excel_files:
    
    a=pd.read_excel(nam)
    
    x=[]
    for b in a.columns[1:]:
        x.append(np.array(a[b])[211:528])
    
    X=np.array(a[a.columns[0]])[211:528]
    
    np.savetxt(nam[:-4]+'csv',x,delimiter=',')
    np.save('Xaxis',X)
    
    
    
np.save('xstn',a['Capture_4:204'][211:528])