#!/usr/bin/env python
# coding: utf-8

# ##This Kernel will follow the next steps:##
# ##1st- Preprocess data##
# ##2nd- Find correlations##
# ##3rd- Linnear regression##

# In[33]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ##1st- Preprocess data##
# 
# 

# In[34]:


#As the data has a date column let's parse the dates for later analisys
df = pd.read_csv('../input/kc_house_data.csv', parse_dates = ['date'])
df.head()


# In[35]:


#Kind of data we are working
df.info()


# There is no needed to use dummy varibales. 

# In[36]:


#Find null values
df.isnull().sum()


# No null variables

# In[37]:


#Extract from 'date of sale' and create two new columns with month an year to find relations with price
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year


# In[38]:


df.head()


# ##2nd- Find Correlations##

# In[39]:


#Correlation
corr = df.corr()

#Listed correlation with the price. 23 is the number of columns of the df
corr.nlargest(23, 'price')['price']


# Closer to 1 or -1 more correlated the values.  Closer to 0 the less correlated, in this case the year is the less correlated with the price. Reading those values there is nothing strange with them.

# In[40]:



#Mybe there is not correlation between dates and price but the prices can be stationary
#I will also check if the prices had increase from 2014 to 2015

#Price increase between 2014 adn 2015
priceYear =  df['price'].groupby(df['year']).mean()
priceYear.plot(kind = 'bar')


# In[41]:


#% value of the icrease
list_priceYear = list(priceYear)
priceIncrease = ((list_priceYear[0]/list_priceYear[1])-1)*(-100)
print ('Form 2014 to 2015 there is a price increase in % of: ', priceIncrease)


# In[42]:


#Find if the prices are stationary between months
priceMonth = df['price'].groupby(df['month']).mean()
priceMonth.plot(kind = 'line')


# **If you buy a house the best month is February with the lower average prices!!!** 

# In[43]:


#The difference in the average price between February and May
print('The average price diference in $ buying a house in Feb or in May is: ', priceMonth.max()-priceMonth.min())


# Seems  that in King Country they like he sun. Nearly 54000$ At least when they buy houses.

# ##3rd- Linear regression##

# In[44]:


#Create the data to train.
y = df['price']
df = df.drop(['price', 'id', 'date'], axis = 1)
x_train,x_test,y_train,y_test=train_test_split(df,y,train_size=0.8,random_state=42)


# In[45]:


#Linnear regression
reg=LinearRegression()
reg.fit(x_train,y_train)
reg.score(x_test,y_test)


# 70% is not a bad result for starting. There are much more powerful algorithms to  make this prediction.
