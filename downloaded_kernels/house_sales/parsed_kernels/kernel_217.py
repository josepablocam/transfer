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
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectPercentile

print((os.listdir("../input")))

# Any results you write to the current directory are saved as output.


# In[2]:


df = pd.read_csv('../input/kc_house_data.csv')
df = pd.get_dummies(df, 'zipcode')
df.head(5)


# # Visualization of different distributions of features in King county

# In[3]:


def plot_cats(cat):
    baths = df[cat].unique()
    baths.sort()
    total_baths = [df[df[cat] == b][cat].count() for b in baths]
    plt.bar(baths, total_baths)


# In[4]:


plot_cats('grade')


# ## Year built distribution of the saled houses

# In[5]:


plot_cats('yr_built')


# In[6]:


df.corr()


# In[7]:


df['yr_renovated'] = df.loc[df.yr_renovated > 2007 ,'yr_renovated'] = 1
df['yr_renovated'] = df.loc[df.yr_renovated <= 2007 ,'yr_renovated'] = 0
df = df.rename(columns = {"yr_renovated" : "is_renovated_in_last_10_years"})


# In[20]:


df.to_csv('ready.csv')


# So we can see that bathrooms, living, grade, sqft_above, sqft_living15  are in great correlation to the price.  
# Let's convert zipcode to be categorical feature

# In[8]:


# features_df = df[['sqft_living','bathrooms', 'sqft_living15', 'grade', 'bedrooms', 'floors', 'waterfront', \
#                   'view', 'sqft_above', 'sqft_basement', 'sqft_lot15', 'lat', 'is_renovated_in_last_10_years']]
features_df = df.drop('price', axis=1)
features_df = SelectPercentile(percentile = 75).fit(features_df,df.price).transform(features_df)
features_df = StandardScaler().fit(features_df).transform(features_df)


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(features_df, df.price)


# In[15]:


linear_regr =  linear_model.ElasticNet(alpha=0.001, max_iter = 5000) # RandomForestRegressor(n_estimators = 75) #
model = linear_regr.fit(x_train, y_train)
predictions = model.predict(x_test)


# In[17]:


plt.scatter(y_test,predictions)
plt.rcParams["figure.figsize"] =(15,12)
plt.show()


# In[19]:


print(("Mean squared error: %.3f"% mean_squared_error(y_test, predictions)))
print(("Mean absolute error: %.3f"% mean_absolute_error(y_test, predictions)))
print(('Variance score: %.3f' % r2_score(y_test, predictions)))


# In[14]:


approx = linear_model.LinearRegression().fit(pd.DataFrame(y_test),predictions).predict(pd.DataFrame(y_test))
plt.plot(y_test,approx)
plt.plot(np.arange(8000000), np.arange(8000000))
plt.show()

