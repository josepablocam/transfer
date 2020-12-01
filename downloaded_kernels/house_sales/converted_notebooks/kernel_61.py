#!/usr/bin/env python
# coding: utf-8

# # Machine Learning on Houses Data
# Started on 19 April 2018
# * Here, I am tinkering with applying machine learning models on houses datasets.
# * Still work-in-progress

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score


# ## Load data

# In[2]:


melbourne_data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')
melbourne_data.head()


# ## Explore data

# In[7]:


melbourne_data.columns


# In[8]:


melbourne_data.describe()


# `Price` would be the target variables. Beside 'id' or `Unnamed: 0` column, the other columns would be potential features for ML models.

# ## First ML model

# In[9]:


# Choose the prediction targets

y = melbourne_data.Price


# In[11]:


# Choose the predictors. Let's start with a few columns which are numeric.

melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']


# In[12]:


X = melbourne_data[melbourne_predictors]


# X.shape

# In[ ]:


sns.set()
house_df.hist(figsize=(15,15))
plt.show()


# In[ ]:


# collect all the numerical data
cols_numeric = ['price', 'bathrooms', 'bedrooms', 'sqft_living', 'sqft_lot', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15',
               'floors']


# In[ ]:


numeric_df = house_df[cols_numeric]


# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Correlation of Features', y=1.05, size=15)
sns.heatmap(numeric_df.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, 
            linecolor='white', annot=True)


# In[ ]:


plt.figure()
plt.scatter(house_df['floors'],house_df['price'])
plt.show()


# ## Split data into training and test sets

# In[ ]:


# transform price to log scale
y = np.log1p(house_df.price)


# In[ ]:


cols_to_fit = ['sqft_living','bedrooms','bathrooms','floors','condition','zipcode']
X = house_df[cols_to_fit]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=123)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


# ## Data preprocessing

# In[ ]:


scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))


# In[ ]:


X_test_scaled = scaler.transform(X_test)
print(X_test_scaled.mean(axis=0))
print(X_test_scaled.std(axis=0))


# ## Use Linear Regression model to train a prediction model

# In[ ]:


lr = LinearRegression().fit(X_train,y_train)
lr


# In[ ]:


y_pred_lr = lr.predict(X_test)
y_pred_lr


# In[ ]:


print(mean_squared_error(y_test,y_pred_lr),r2_score(y_test,y_pred_lr))


# ## Use Ridge Regression to train a prediction model

# In[ ]:


ridge = Ridge()
ridge


# In[ ]:


grid_values = {'alpha':[50,100,200,300,400,500]}
grid_ridge = GridSearchCV(ridge,param_grid=grid_values,cv=10)
grid_ridge.fit(X_train,y_train)


# In[ ]:


print(grid_ridge.best_params_)


# In[ ]:


ridge = Ridge(alpha=100).fit(X_train,y_train)
ridge


# In[ ]:


y_pred_ridge = ridge.predict(X_test)
print(mean_squared_error(y_test,y_pred_ridge),r2_score(y_test,y_pred_ridge))


# ## Use Random Forest Regressor to train a prediction model

# In[ ]:


rf = RandomForestRegressor()
rf


# In[ ]:


grid_values = {'max_features':['auto','sqrt','log2'],'max_depth':[None,5,3,1]}
grid_rf = GridSearchCV(rf,param_grid=grid_values,cv=10)
grid_rf.fit(X_train,y_train)


# In[ ]:


print(grid_rf.best_params_)


# In[ ]:


rf = RandomForestRegressor().fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)


# In[ ]:


print(mean_squared_error(y_test,y_pred_rf),r2_score(y_test,y_pred_rf))


# ## Use Lasso Regression to train a prediction model

# In[ ]:


lasso = Lasso()
lasso


# In[ ]:


grid_values = {'alpha':[0.00001,0.0001,0.001, 0.01,0.1,1],'max_iter':[1000,10000]}
grid_lasso = GridSearchCV(lasso,param_grid=grid_values,cv=10)
grid_lasso.fit(X_train,y_train)


# In[ ]:


print(grid_lasso.best_params_)


# In[ ]:


lasso = Lasso(alpha=0.0001).fit(X_train,y_train)
y_pred_lasso = lasso.predict(X_test)
print(mean_squared_error(y_test,y_pred_lasso),r2_score(y_test,y_pred_lasso))


# ### Random Forest Regressor has the lowest mean squared error and highest r2 score among the models I fitted.
