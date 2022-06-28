#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 07:26:26 2022

@author: elhamebr
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use("Agg")

ct=np.load('ct.npy')
cv=np.load('cv.npy')


x=np.load('xdl.npy')



for i in range(2900):
    fig,ax=plt.subplots(3,2,figsize=(22/2,18/2),
                        gridspec_kw={'height_ratios': [6,1,1]})

    gs = ax[1, 0].get_gridspec()
    ax[1,0].remove()
    ax[1,1].remove()
    ax[2,0].remove()
    ax[2,1].remove()

    axbig1 = fig.add_subplot(gs[1, :])
    axbig2 = fig.add_subplot(gs[2, :])


    a0=np.load('train%i.npy'%(i+2))
    a1=np.load('val%i.npy'%(i+2))
    a2=np.load('lat%i.npy'%(i+2))
    ax[0,0].scatter(a0.T[0],a0.T[1],c=ct,cmap='jet')
    ax[0,1].scatter(a1.T[0],a1.T[1],c=cv,cmap='jet')
    
    ax[0,0].set_xlabel('X',fontweight='bold')
    ax[0,0].set_ylabel('Y',fontweight='bold')

    
    
    ax[0,1].set_xlabel('X',fontweight='bold')
    ax[0,1].set_ylabel('Y',fontweight='bold')

    
    ax[0,0].set_title('Training set',fontweight='bold')
    ax[0,1].set_title('Test set',fontweight='bold')
    
    
    
    
    axbig1.plot(x,a2.T[0][1:])
    axbig2.plot(x,a2.T[1][1:])

    axbig1.set_title('X score',fontweight='bold')
    axbig2.set_title('Y score',fontweight='bold')

    axbig1.set_xticklabels([])
    axbig2.set_xlabel('Raman Shift $(cm^{-1})$')
    
    for axs in [axbig1,axbig2]: 
        axs.axvspan(850, 950, alpha=0.4, color='#94a2b3')    #protoin lipid
        axs.axvspan(990, 1010, alpha=0.4, color='purple')     #phenililanine
        axs.axvspan(1050, 1150, alpha=0.4, color='#deb4db')    #dna protoin
        axs.axvspan(1200, 1280, alpha=0.4, color='#e8d190')    #protoin
        axs.axvspan(1300, 1350, alpha=0.4, color='#ade8ff')    #lipid
        axs.axvspan(1410, 1490, alpha=0.4, color='#c75a7f')   #dna lipid protoin
        axs.axvspan(1520, 1600, alpha=0.4, color='#8da9f7')    #dna rna
        axs.axvspan(1610, 1700, alpha=0.4, color='#e8d190')    #protoin
        axs.axvspan(1730, 1760, alpha=0.4, color='#7179c7')     #lipid/Phospholipids
        axs.axvspan(657, 690, alpha=0.4, color='#8da9f7')   #dna rna
        axs.axvspan(752, 812, alpha=0.4, color='#8da9f7')   #dna rna
    
    plt.tight_layout()
    plt.savefig('./1/%i.png'%i)
    #plt.show()
    print(i)