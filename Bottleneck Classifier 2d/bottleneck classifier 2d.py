#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 06:25:02 2022

@author: elhamebr
"""


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten , AveragePooling1D, MaxPooling1D, BatchNormalization,\
Activation
model = Sequential()
import matplotlib.pyplot as plt
import seaborn as sns




#### Necessary Imports for Neural Net 

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding1D,\
     Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add, Reshape 
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

import numpy as np



import pandas as pd



name=['LPS-P.aeruginosa.csv','Pyocynin.csv','PGN-SA.csv','LTA-SA.csv','LPS-KPS.csv']

tst=[]
test=[0.0 for i in range(1000)]
test=np.array(test)
tst.append(test)

for i in range(1000):
    test[i]=1
    tst.append(test)
    test=test*0

tst=np.array(tst)

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



X = cdl.drop('type',axis=1)
y = cdl['type']





import numpy as np
X=np.array(X)
y=np.array(y)



    

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.5,random_state=20)


ct=y_train
cv=y_val


np.save('./latent data batch/ct',ct)
np.save('./latent data batch/cv',cv)


c_train=y_train
c_val=y_val



from tensorflow.keras.utils import to_categorical
#one-hot encode target column
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)







def resnet50():

  input_im = Input(shape=(len(X_train[0]))) # cifar 10 images size

  x = Dense(800)(input_im)
  x = Dense(400)(x)

  
  x0 = Dense(2,name='la')(x) #multi-class
  

  #Change the x1 based on the amount of data here
  xint = Dense(100,activation='relu')(x0)
  xint = Dense(100,activation='relu')(xint)
  x1 = Dense(5,activation='softmax')(xint)
  
  
  

  # define the model 

  model = Model(inputs=input_im, outputs=[x1], name='Resnet50')

  return model

resnet50_model = resnet50()


import keras as keras



resnet50_model.compile(loss=['CategoricalCrossentropy'], optimizer=Adam(learning_rate=1e-3),metrics=['acc'])

il=Model(inputs=resnet50_model.input,outputs=resnet50_model.get_layer('la').output)



X=X.reshape(X.shape[0],len(X[0]),1)

import tensorflow as tf


       
jj=1  
from random import uniform
class TestCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
        
        
#         
#         # if jj%1==0:
#         #     a0=il.predict(X_train)
#         #     a1=il.predict(X_val)
#         #     a2=il.predict(tst)
#         #     np.save('./latent data/train%i'%jj,a0)
#         #     np.save('./latent data/val%i'%jj,a1)
#         #     np.save('./latent data/lat%i'%jj,a2)
            
    def on_train_batch_end(self, batch, epoch, logs=None):
        global jj
        jj=jj+1
        if jj%1==0:
            
            a0=il.predict(X_train)
            a1=il.predict(X_val)
            a2=il.predict(tst)
            np.save('./latent data batch/train%i'%jj,a0)
            np.save('./latent data batch/val%i'%jj,a1)
            np.save('./latent data batch/lat%i'%jj,a2)

        
            
        
        

        


his=resnet50_model.fit(X_train,y_train, validation_data=(X_val, y_val),
                            epochs=100, callbacks=[TestCallback()],batch_size=100)


X=[]

test=[0.0 for i in range(len(X_train[0]))]
M=il.predict([test])[0]
testM=[]
for i in range(len(X_train[0])):
    test=[0.0 for i in range(len(X_train[0]))]
    test[i]=1
    testM.append(test)

a=il.predict(testM)

for i in range(len(X_train[0])):
    X.append(a[i][0]-M[0])

    
    
    
    
    