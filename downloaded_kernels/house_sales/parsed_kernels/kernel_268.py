#!/usr/bin/env python
# coding: utf-8

# **Linear regression with multiple features**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print((check_output(["ls", "../input"]).decode("utf8")))

# Any results you write to the current directory are saved as output.


# **Step 1 : Loading the house data into a dataframe and determining the data types**

# In[ ]:


dataFrame = pd.read_csv('../input/kc_house_data.csv')
print(("Total records in the sample : %d" % len(dataFrame)))
print((dataFrame.columns.values))
dataFrame.dtypes


# **Step 2: From dataframe get interested  features x  to  predict y - house price **

# In[ ]:


interested_features = ['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors','waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode']
x = dataFrame[interested_features]
y = dataFrame["price"]


# **Step 3 : Split the data into training and test set**

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.30, random_state=3)


# **Step 4: Fit the data into linear regression**

# In[ ]:


regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
print(("No. of rows considered as training : %d " % len(x_train)))
print(("No. of rows considered as test : %d " % len(x_test)))

x_train.shape
y_train.shape


# **Step 5 : Calculate the accuracy **

# In[ ]:


accuracy = regr.score(x_test,y_test)  # .score(x_test, y_test)
print(("Accuracy achieved : {}%".format(int(round(accuracy * 100)))))


# In[ ]:




