#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print((check_output(["ls", "../input"]).decode("utf8")))

# Any results you write to the current directory are saved as output.


# In[4]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[5]:


import os
import time

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl

import numpy as np
np.set_printoptions(precision=2, linewidth=120, suppress=True, edgeitems=4)

import pandas as pd
pd.set_option('display.max_columns', 150)
#pd.set_option('precision', 5)

from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor,RandomizedLasso
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


# In[7]:


os.getcwd()
get_ipython().system('ls')


# In[9]:


os.chdir('../input/')


# In[10]:


KING = pd.read_csv('kc_house_data.csv',sep=',',error_bad_lines=False,parse_dates=['date'])


# In[11]:


KING.head(4)


# In[32]:


# Select our independent variables to feed into the regression model as 'X'
X = KING.ix[:,np.r_[3:17,19:21]] # Let's remove the latitude and longitude but keep zip as categorical
X.head(4)

# We can remove these two columns as predictors and save for later to link up to the predictions
index = KING.ix[:,np.r_[0:2]]
index.head(4)


# In[42]:


X['key'] = 'key'


# In[34]:


# Add in a column of zeros as the intercept for multiple linear regression

#pd.DataFrame(np.ones(len(X)))

pd.merge(X,pd.DataFrame(np.ones(len(X))))


# In[36]:


# standardize the numeric independent predictor variables 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
X_s = scaler.transform(X)


# In[37]:


X_s


# In[38]:


# Select the price column as our dependent target variable
Y = KING.ix[:,2]
Y.head(4)


# In[ ]:





# In[ ]:




