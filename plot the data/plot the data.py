#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 03:29:03 2022

@author: elhamebr
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import uniform
import seaborn as sns


name=['LPS-P.aeruginosa.csv','Pyocynin.csv','PGN-SA.csv','LTA-SA.csv','LPS-KPS.csv']


namestn=[]
for i in range(len(name)):
    namestn.append(name[i][:-4]+'-stn.csv')


namedl=[]
for i in range(len(name)):
    namedl.append(name[i][:-4]+'-pre.csv')
    
    
    
a=pd.read_csv(name[0],sep='\t')
xstn=np.load('xstn.npy')
xdl=np.interp(range(1000),np.linspace(0,999,len(xstn)),xstn)
np.save('xdl',xdl)


fig,ax=plt.subplots(15,2,gridspec_kw={'height_ratios': [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]},
                    figsize=(14*1.5,10*1.5),sharex='col')

gs = ax[0, 0].get_gridspec()
ax[0,0].remove()
ax[0,1].remove()
axbig = fig.add_subplot(gs[0, :])


for i in range(len(namestn)):
    dataset1 = pd.read_csv(namestn[i],header=None)
    a=np.array(dataset1)
    ax[i+1,0].plot(xstn,a.mean(axis=0),label='mean of '+namestn[i][:-8])
    ax[i+1,0].fill_between(xstn,a.mean(axis=0)-a.std(axis=0)/2,a.mean(axis=0)+a.std(axis=0)/2,
                           color='r',alpha=.5,label='std')
    ax[i+1,0].legend(fontsize=7.5,loc=2)
    

for i in range(len(namedl)):
    dataset1 = pd.read_csv(namedl[i],header=None)
    a=np.array(dataset1)
    ax[i+1,1].plot(xdl,a.mean(axis=0),label='mean of '+namestn[i][:-8])
    ax[i+1,1].fill_between(xdl,a.mean(axis=0)-a.std(axis=0)/2,a.mean(axis=0)+a.std(axis=0)/2,
                           color='r',alpha=.5,label='std')
    ax[i+1,1].legend(fontsize=7.5,loc=2)
    
ax[1,0].set_title('raw Data',fontweight='bold')
ax[1,1].set_title('Deep learning pre-processing',fontweight='bold')

axis=[]
for i in range(5):
    for j in range(2):
        axis.append(ax[i+1,j])



for axs in axis: 
    axs.set_yticks([])
for axs in axis: 
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
    


colors=[['#94a2b3','purple','#deb4db','#e8d190','#ade8ff','#c75a7f','#8da9f7','#7179c7']]

columns=[["protein/lipid","phenylalanine","DNA/protein","protein"],
         ["lipid", "DNA/lipid/protein","DNA/RNA","lipid/phospholipids"]]

tabledata=[['#94a2b3','purple','#deb4db','#e8d190'],
           ['#ade8ff','#c75a7f','#8da9f7','#7179c7']]

the_table=axbig.table(cellColours=tabledata,
                      cellText=columns,
                      loc='bottom', fontsize='bold')
the_table.auto_set_font_size(False)
# the_table.set_fontsize(7)

axbig.axis('off')

for cell in the_table._cells:
    the_table._cells[cell].set_alpha(.4)


axbig.set_position([.15,.89,.7,.05])
# ax[1].set_xticks([800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800])
ax[-1,1].set_xlabel('Raman shift ($cm^{-1}$)')
ax[-1,0].set_xlabel('Raman shift ($cm^{-1}$)')
# ax[1].grid()

plt.savefig('score-latent.pdf')
plt.show()








