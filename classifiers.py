#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 05:39:53 2022

@author: elhamebr
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 00:53:04 2021

@author: mohammadkazemzadeh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from random import sample, uniform

name=['LPS-P.aeruginosa.csv','Pyocynin.csv','PGN-SA.csv','LTA-SA.csv','LPS-KPS.csv']



namestn=[]
for i in range(len(name)):
    namestn.append(name[i][:-4]+'-stn.csv')


namedl=[]
for i in range(len(name)):
    namedl.append(name[i][:-4]+'-pre.csv')
    
    
dataset1 = pd.read_csv(namedl[0],header=None)
dataset1["type"]=0
cdl=pd.concat([dataset1],axis=0,join='inner')
cdl=cdl.reset_index(drop=True)

t=1
for file in namedl[1:]:
    dataset1 = pd.read_csv(file,header=None)
    dataset1["type"]=t
    cdl=pd.concat([cdl,dataset1],axis=0,join='inner')
    cdl=cdl.reset_index(drop=True)
    t=t+1
   



dataset1 = pd.read_csv(namestn[0],header=None)
dataset1["type"]=0
cstn=pd.concat([dataset1],axis=0,join='inner')
cstn=cstn.reset_index(drop=True)

t=1
for file in namestn[1:]:
    dataset1 = pd.read_csv(file,header=None)
    dataset1["type"]=t
    cstn=pd.concat([cstn,dataset1],axis=0,join='inner')
    cstn=cstn.reset_index(drop=True)
    t=t+1



for c,nam in zip([cstn,cdl],['stn','dl']):
        
    c=c.reset_index(drop=True)
    
    from sklearn.model_selection import train_test_split
    
    X = c.drop('type',axis=1)
    y = c['type']
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.5,random_state=21)
    


    
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    
    namm=["LDA","SVM","RF","ANN","KNN"]
    
    for name in namm:
        print(name)
        
        
        if name=="LDA":
        
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            clflda = LinearDiscriminantAnalysis()
            clflda.fit(X_train, y_train)
    
        
        
        
        if name=="RF":
            
            from sklearn.ensemble import RandomForestClassifier
            clfrf = RandomForestClassifier(max_depth=40)
            clfrf.fit(X_train, y_train)
    
        
        if name=="ANN":
            
            from sklearn.neural_network import MLPClassifier
            clfann = MLPClassifier(hidden_layer_sizes=(1000,300,60),alpha=1, activation='relu', solver='adam', max_iter=5000)
            clfann.fit(X_train,y_train)
    
    
        
        
        if name=="SVM":
            
            from sklearn.svm import SVC
            clfsvm=SVC(kernel="rbf")
            clfsvm.fit(X_train,y_train)
    
        
        if name=="GPC":
            
            from sklearn.gaussian_process import GaussianProcessClassifier
            from sklearn.gaussian_process.kernels import RBF 
            clfgpc=GaussianProcessClassifier(1 * RBF(7))
            clfgpc.fit(X_train,y_train)
    
        
        if name=="KNN":
        
            from sklearn.neighbors import KNeighborsClassifier
            clfknn=KNeighborsClassifier(10)
            clfknn.fit(X_train,y_train)
    
        
        
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    
    


    
    acc=[]
    acc_ml=[]
    
    for clf,qq in zip([clflda,clfsvm,clfrf,clfann,clfknn],[1,2,3,4,5]):
        
        y_predict=clf.predict(X_test)
        acc.append(accuracy_score(y_test,y_predict))
        print(accuracy_score(y_test,y_predict))
    np.savetxt('acc '+nam+'.txt',acc,delimiter=',')
    acc=[]
    
