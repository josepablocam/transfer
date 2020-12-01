#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print((os.listdir("../input")))

# Any results you write to the current directory are saved as output.


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, GridSearchCV
from IPython.display import display
import seaborn as sns
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv("../input/kc_house_data.csv")

X = df[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_basement', 'yr_built', 'zipcode',
       'sqft_living15', 'sqft_lot15']]

y = df['price']

d = pd.get_dummies(df['zipcode'].astype('str'))
X = pd.concat([X,d],axis=1)
X = X.drop('zipcode',axis=1)


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

X_train,X_val,y_train,y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=1234)

from sklearn import preprocessing

scalerX = preprocessing.StandardScaler()
X_train = scalerX.fit_transform(X_train)
scalerY = preprocessing.StandardScaler()
y_train = scalerY.fit_transform(np.log(y_train).reshape(-1,1))

X_val = scalerX.transform(X_val)
y_val = scalerY.transform(np.log(y_val).reshape(-1,1))

X_test = scalerX.transform(X_test)
y_test = scalerY.transform(np.log(y_test).reshape(-1,1))

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD,RMSprop, Adagrad, Adadelta, Adam

input_dim = len(X.columns)

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=input_dim))
model.add(Dense(64, activation='relu', input_dim=128))
model.add(Dense(32, activation='relu', input_dim=64))

model.add(Dense(1))

sgd = SGD(lr=5e-3,nesterov=False)
#rms = RMSprop(lr=0.01)
#adag = Adagrad(lr=0.01)
#adad = Adadelta(lr=0.01)
#adam = Adam(lr=0.01)

model.compile(loss='mean_absolute_error',
              optimizer=sgd,
              metrics=['mean_absolute_error'])


# In[10]:


fit = model.fit(X_train, y_train,
          epochs=500,
          batch_size=512,validation_data=(X_val, y_val),verbose=0)

df = pd.DataFrame(fit.history)

df[["loss", "val_loss"]].plot()
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()


# In[11]:


y_true = np.exp(scalerY.inverse_transform(y_test)).reshape(-1)
y_pred = np.exp(scalerY.inverse_transform(fit.model.predict(X_test))).reshape(-1)
np.abs(y_pred-y_true).mean()


# In[ ]:




