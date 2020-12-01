#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[7]:


import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

df = pd.read_csv("../input/kc_house_data.csv")


# In[8]:


display(df.head())
display(df.tail())


# In[9]:


print((df.info()))


# No missing value.

# In[10]:


print(("Data shape: {}" .format(df.shape)))


# Number of data points: 21,613 <br>
# Number of feature quantities: 21

# In[11]:


df.describe()


# In[12]:


#Take a look at the heat map with seaborn
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[13]:


#Another way to look at the heat map 
df.corr().style.background_gradient().format('{:.2f}')


# In[14]:


#Check price and the scatter plot of each feature
df2 = df.drop(['zipcode','id'],axis=1)

for c in df2.columns:
    if (c != 'price') & (c != 'date'):
        df2[[c,'price']].plot(kind='scatter',x=c,y='price')


# ### Selection of feature quantity (1)
# 
# - Let's pick up a characteristic amount with high correlation with price
# sqft_living, grade, sqft_above
# - The correlation of sqft_living 15 is also high, but when looking at the description of data (https://www.kaggle.com/harlfoxem/housesalesprediction/data), Living room area in 2015 (implies - some renovations)
# - This is or might not have affected the lotsize area and is excluded because it is old in the 2015 sqft_living data
# - Exclude sqft_lot 15 for the same reason as sqft_living 15

# In[15]:


df.date.head()


# In[16]:


#date conversion
pd.to_datetime(df.date).head()


# In[23]:


#df_en_fin = df.drop(['date','zipcode','sqft_living15','sqft_lot15'],axis=1)


# ### Check results of each algorithm and select algorithm
# Confirm results with the following algorithm
# 
# 1. Linear regression
# 1. Random Forest
# 1. gradient boosting
# 1. k neighborhood method

# In[18]:


#1.Linear regression

df = pd.read_csv("../input/kc_house_data.csv")
X = df.drop(["id", "price", "zipcode", "date"], axis=1)
y = df["price"]

regr = LinearRegression()
scores = cross_val_score(regr, X, y, cv=10)
print(("score: %s"%scores.mean()))


# In[19]:


#2.Random Forest

df = pd.read_csv("../input/kc_house_data.csv")
X = df.drop(["id", "price", "zipcode", "date"], axis=1)
y = df["price"]

regr = RandomForestRegressor()
scores = cross_val_score(regr, X, y, cv=10)
print(("score: %s"%scores.mean()))


# In[20]:


#3.gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("../input/kc_house_data.csv")
X = df.drop(["id", "price", "zipcode", "date"], axis=1)
y = df["price"]

gbrt = GradientBoostingClassifier()
scores = cross_val_score(regr, X, y, cv=10)
print(("score: %s"%scores.mean()))


# In[22]:


#4.k neighborhood method
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("../input/kc_house_data.csv")
X = df.drop(["id", "price", "zipcode", "date"], axis=1)
y = df["price"]

n_neighbors = KNeighborsClassifier()
scores = cross_val_score(regr, X, y, cv=10)
print(("score: %s"%scores.mean()))


# ### Conclusion so far and what to do in the future
# - The highest score was random forest followed by the k-nearest neighbor method
# - Evaluate prediction performance with two algorithms of k-neighbor method and random forest

# In[ ]:




