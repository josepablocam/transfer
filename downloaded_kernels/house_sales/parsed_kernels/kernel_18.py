#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats #to call a function that removes anomalies

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print((check_output(["ls", "../input"]).decode("utf8")))

# Any results you write to the current directory are saved as output.


# Hello,
# 
# So I analysed certain factors to see if they had any relationships with house prices and the factors that had the most relationships were number of bathrooms, grade and sqft_living. 
# 
# The coefficient result was quite interesting and unexpected, you should definitely check it out.
# 
# I'm still new at this and soo all feedback is greatly appreciated.
# 
# Cheers!
# 
# Fayomi

# In[ ]:





# In[ ]:


df = pd.read_csv('../input/kc_house_data.csv')


# In[ ]:


df.head()


# In[ ]:


df.drop(['id','date','sqft_lot','sqft_above','lat', 'long','zipcode', 'sqft_living15', 'sqft_lot15','waterfront','view'],axis=1,inplace=True)


# In[ ]:


df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)] #to remove anomalies
df.head()


# In[ ]:


df.info()


# In[ ]:





# In[ ]:


plt.figure(figsize=(16,6))
sns.distplot(df['price'],kde=False,bins=50)


# In[ ]:





# In[ ]:


plt.figure(figsize=(16,6))
sns.distplot(df['price'].dropna(),kde=False,bins=50)


# In[ ]:


plt.figure(figsize=(16,6))
sns.countplot(df['bedrooms'])


# In[ ]:


plt.figure(figsize=(16,6))
sns.countplot(df['bathrooms'])


# In[ ]:


plt.figure(figsize=(16,6))
sns.distplot(df['sqft_living'].dropna(),kde=False,bins=50)


# In[ ]:


sns.pairplot(df)


# In[ ]:


sns.jointplot(x='bedrooms',y='price',data=df)


# In[ ]:


sns.jointplot(x='price',y='sqft_living',data=df,kind='reg')


# In[ ]:


sns.jointplot(x='floors',y='price',data=df)


# In[ ]:


sns.jointplot(x='grade',y='price',data=df, kind='reg')


# In[ ]:


sns.jointplot(x='yr_built',y='price',data=df)


# In[ ]:


sns.jointplot(x='sqft_basement',y='price',data=df)


# In[ ]:





# In[ ]:


sns.jointplot(x='bathrooms',y='price',data=df, kind='reg')


# In[ ]:


sns.jointplot(x='condition',y='price',data=df)


# the conditions most correlated with price are: bathrooms,, grade, sqft_living (and maybe bedrooms)

# In[ ]:


sns.heatmap(df.corr(),cmap='coolwarm', annot=True)


# 
# TIME TO FORMAT DATA FOR ML

# In[ ]:



df.columns


# In[ ]:


#selected inputs
x = df[['bathrooms','grade','sqft_living']]
#expected output
y = df['price']


# In[ ]:


from sklearn.cross_validation import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


#to train the data
lm.fit(x_train,y_train)


# In[ ]:


#to calculate teh coefficients
lm.coef_


# In[ ]:


#to create a table with the coefs
cdf = pd.DataFrame(lm.coef_,x.columns,columns=['coefs'])


# In[ ]:


cdf


# In[ ]:


#to get the predictions of test set
pred = lm.predict(x_test)


# In[ ]:


#to plot predictions and actual result
#This shows an accurate preditction
plt.scatter(y_test, pred)


# In[ ]:





# In[ ]:





# In[ ]:




