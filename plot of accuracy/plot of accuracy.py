#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 08:24:36 2022

@author: elhamebr
"""

import matplotlib.pyplot as plt
import numpy as np

fig,ax=plt.subplots(1,figsize=(8,4))
x=[0,1,2,3,4]
#These are Preprocessed Data from acc dl.txt
#y= [9.992887624466572083e-01,
#9.982219061166429652e-01,
#9.989331436699857569e-01,
#9.989331436699857569e-01,
#9.957325746799431387e-01]


#These are raw data from acc stn.txt
y= [9.964438122332859304e-01,
9.985775248933144166e-01,
9.960881934566144791e-01,
9.978662873399715139e-01,
9.740398293029871590e-01]


c=['#8f0000','#263c8c','#8f0000','#263c8c','#8f0000']
y=np.array(y)
#Dont forget to change Z value
z=.964
ax.grid(zorder=0)
ax.bar(x,y-z, bottom=z, zorder=3,color=c)

ax.set_xticks(x)

ax.set_xticklabels([ 'LDA', 'SVM', "RF",'ANN',"KNN"])

ax.set_title('accuracy comparison raw data',fontweight='bold')
ax.set_ylabel('Accuracy',fontsize=14)
ax.set_xlabel('Models',fontsize=14)


plt.savefig('MC_raw.pdf')