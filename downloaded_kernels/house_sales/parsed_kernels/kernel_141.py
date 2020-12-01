#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime,date,time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print((check_output(["ls", "../input"]).decode("utf8")))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("../input/kc_house_data.csv")
data.head()


# In[ ]:


data.shape# find number of rows and columns 


# Find any NaN/Null values

# In[ ]:


data.isnull().sum() 


# The data is clean with no NaN or Null values for all features

# In[ ]:


data.info()


# Using datetime extract the month and year

# In[ ]:


data["date"]=pd.to_datetime(data["date"])
data["month"]=data["date"].dt.month
data["year"]=data['date'].dt.year


# calculate the age of house

# In[ ]:


current_year=datetime.now().year
data["house_age"]=current_year-data["yr_built"] #create new colums for house_age


# In[ ]:


data=data.drop(["id",'date'],axis=1)
data.head()


# **Distributionplot for Price**

# In[ ]:


sns.distplot(data.price)


# In[ ]:


a=data.ix[data["year"]==2015]['month'].value_counts()
a


# In[ ]:


b=data.ix[data["year"]==2014]['month'].value_counts()
b


# In[ ]:


a=data.groupby(["year",'month'])["month"].count().unstack("year")
ax = a.plot(kind='bar', stacked=True, alpha=0.7)
ax.set_xlabel('month', fontsize=14)
ax.set_ylabel('count', fontsize=14)
plt.xticks(rotation=0)
plt.show()


# **month vs price distribution**

# In[ ]:


price_month=data['price'].groupby(data['month']).mean()
price_month.plot(kind='line')
plt.show()


# Above plot shows that, in  February month price is  low, in April  price is very high.
# so,  you can buy a house in February month with best price. 

# In[ ]:


#price difference between february & April
price_difference=price_month.max()-price_month.min()
price_difference


# **Correlation**

# In[ ]:


corr=data.corr()
corr.nlargest(24,'price')['price']


# Target is price,so we can obsearve that, 
# 1. sqft_living,grade,sqft_above,sqft_living15 is more correlated with price.
# 2.year and price is no correlation

# Using heatmap to obsearve the correlation between the features 

# In[ ]:


#df correlation matrix
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(corr, annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()


# In[ ]:


labels=['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated','lat', 'long', 'sqft_living15', 'sqft_lot15']
for i in range(len(labels)):
    plt.figure()
    sns.regplot(x=data[labels[i]],y="price",data=data);
    plt.xlabel(labels[i])
    plt.ylabel('price')
    plt.show()


# In[ ]:


X=data[['sqft_living','sqft_above','house_age',
        'lat', 'long', 'sqft_living15','zipcode','sqft_lot15','waterfront', 'condition', 'grade']]
Y=data[['price']]


# In[ ]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)


# In[ ]:


Y_train.shape


# **Multiple linear regression**

# In[ ]:


model = LinearRegression()
model.fit(X_train, Y_train)
train_score=model.score(X_train,Y_train)
train_score


# In[ ]:


test_score=model.score(X_test,Y_test)
test_score


# **BaggingRegressor**

# In[ ]:


#Decisiontree Regressor,BaggingRegressor
model = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=10), n_estimators=5,bootstrap=True, bootstrap_features=False, oob_score=True, random_state=2, verbose=1).fit(X_train, Y_train)
test_score=model.score(X_test,Y_test)
train_score=model.score(X_train,Y_train)
train_score


# In[ ]:


test_score


# _Cross validation is use for model performence on unseen data 

# In[ ]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(model,X,Y,cv=2)
score


# **BaggingRegressor using RandomForest**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(max_depth=6,random_state=5)
model.fit(X_train,Y_train)
predict=model.predict(X_test)
predict


# In[ ]:


score=model.score(X_train,Y_train)
score


# In[ ]:


score=model.score(X_test,Y_test)
score

