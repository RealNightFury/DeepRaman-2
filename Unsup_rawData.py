#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 03:03:37 2020

@author: mohammadkazemzadeh
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 22:36:32 2020

@author: mohammadkazemzadeh
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


name=['LPS-P.aeruginosa.csv','Pyocynin.csv','PGN-SA.csv','LTA-SA.csv','LPS-KPS.csv']

# for i in range(len(name)):
#     name[i]=name[i][:-4]+'-pre.csv'
    

for i in range(len(name)):
    name[i]=name[i][:-4]+'-stn.csv'


dataset1 = pd.read_csv(name[0],header=None)
dataset1["type"]=0
c=pd.concat([dataset1],axis=0,join='inner')
c=c.reset_index(drop=True)

t=1
for file in name[1:]:
    dataset1 = pd.read_csv(file,header=None)
    dataset1["type"]=t
    c=pd.concat([c,dataset1],axis=0,join='inner')
    c=c.reset_index(drop=True)
    t=t+1




X = c.drop('type',1)
y = c['type']


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X)


from sklearn.decomposition import PCA
n=20
pca = PCA(n_components=n)
X_train = pca.fit_transform(X_train)

a=pca
# X_test=pca.transform(X_test)


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.xticks(range(n),[i+1 for i in range(n)])
plt.legend()
plt.show()




u=X_train.T[:2]
u=u.T
b=pd.DataFrame(data=u,columns = ['principal component 1', 'principal component 2'])


a=pd.concat([b,y],axis=1)
print(a)


import seaborn as sns


fig,ax=plt.subplots(2,2,figsize=(10,7))

ax[0,0].scatter(b['principal component 1'],b['principal component 2'],c=y,cmap='jet')
















from sklearn.preprocessing import StandardScaler
from sklearn import manifold
import time as time
sc = StandardScaler()
X = sc.fit_transform(X)
from collections import OrderedDict
from functools import partial

n_neighbors = 100
n_components = 2


methods = OrderedDict()

tsne = manifold.TSNE(n_components=n_components, init='pca',
                                   random_state=20,perplexity=10)


emb=tsne.fit_transform(X)

ax[0,1].scatter(emb[:,0],emb[:,1],c=y,cmap='jet')


    

import umap

reducer = umap.UMAP(n_neighbors=100,
        min_dist=.9,
        n_components=2,
        random_state=20,
        metric='euclidean')






scaled_penguin_data = StandardScaler().fit_transform(X)

embedding = reducer.fit_transform(scaled_penguin_data)
print(embedding.shape)



from mpl_toolkits.mplot3d import Axes3D
Axes3D


ax[1,0].scatter(embedding[:, 0],embedding[:, 1], c = y, cmap='jet')




import matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import cm



norm = matplotlib.colors.Normalize(vmin=0, vmax=13)


nam=[]
for i in range(len(name)):
    nam.append(name[i][:-8])

legend_elements=[]


for i in range(len(nam)):
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=nam[i],
                              markerfacecolor=cm.jet(norm(i)), markersize=15))


ax[1,1].legend(handles=legend_elements,ncol=2,fontsize=12.5)

ax[1,1].axis('off')

ax[0,1].set_xlabel('t-SNE 1',fontsize=13)
ax[0,1].set_ylabel('t-SNE 2',fontsize=13)
ax[0,1].set_title('t-SNE',fontweight='bold',fontsize=14)
ax[0,1].grid()


ax[1,0].set_xlabel('UMAP 1',fontsize=13)
ax[1,0].set_ylabel('UMAP 2',fontsize=13)
ax[1,0].set_title('UMAP',fontweight='bold',fontsize=14)
ax[1,0].grid()

ax[0,0].set_xlabel('PCA 1',fontsize=13)
ax[0,0].set_ylabel('PCA 2',fontsize=13)
ax[0,0].set_title('PCA',fontweight='bold',fontsize=14)
ax[0,0].grid()

plt.tight_layout()
plt.savefig('ML-2D-tsne-stn.pdf')
plt.show()






