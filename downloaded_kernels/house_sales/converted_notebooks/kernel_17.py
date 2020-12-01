#!/usr/bin/env python
# coding: utf-8

# # House EDA：The Fast Journey
# 
# `Majin Buu `
# ---
# 
# - **1 First Step**
#     - 1.1 Load libraries and helper functions
#     - 1.2 Load data
#     - 1.3 Check the Memory Usage
#     - 1.4 DataType Converting
# 
# - **2 Univariable Analysis**
# 
# - **3 Bivariate Analysis**
# 
# - **4 Feature Engineer**
# 
# - **5 LightGBM**

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('fivethirtyeight')
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.figsize'] = (10,10)

from scipy.stats import boxcox, norm

import gc
import warnings
warnings.filterwarnings('ignore')
gc.enable()


# In[ ]:


train = pd.read_csv("../input/kc_house_data.csv")
train['date'] = pd.to_datetime(train['date'])
train['date_yr'] = train['date'].dt.year
train.drop('date', axis = 1, inplace = True)
train.head()


# In[ ]:


train.info(verbose=False)


# In[ ]:


for c, dtype in zip(train.columns, train.dtypes):
    if dtype == np.float64:
        train[c] = train[c].astype(np.float32) 
    elif dtype == np.int64:
        train[c] = train[c].astype(np.int32) 


# In[ ]:


train.info(verbose=False)


# In[ ]:


train.columns.values


# ## UniVariable Analyisi

# In[ ]:


train['price'].describe()


# In[ ]:


train['view'].value_counts()


# In[ ]:


train['bedrooms'].value_counts()


# ## Bivariable

# In[ ]:


## Check outlier
train[train['bedrooms']==33]['price']


# ## Feature Engineer

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['zipcode'] = le.fit_transform(train['zipcode'])
train['Life'] = train['date_yr'] - train['yr_built']
train['renovated'] = np.where(train['yr_renovated']!=0 ,1 ,0)
train = train.drop(['id','yr_built','lat','long','date_yr','yr_renovated'], axis = 1)
train['roomcnt'] = train['bedrooms'] + train['bathrooms']
train['sqft_per_room'] = train['sqft_living']/(train['bedrooms'] + train['bathrooms'])
train.head()


# In[ ]:


import xgboost as xg
from xgboost import XGBRegressor
model = XGBRegressor(max_depth = 6, min_child_weight = 10 ,subsample = 0.8 ,colsample_bytree = 0.6
                ,objective = 'reg:linear', num_estimators = 3000 , learning_rate = 0.01)
X= train.drop('price',axis=1)
feat_names = X.columns.values
y = np.log1p(train.price.values)
model.fit(X, y)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title("Feature importances by XGB", fontsize=20) 
plt.bar(range(len(indices)), importances[indices], color='lightblue', align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show();


# In[ ]:


plt.figure()
sns.jointplot(x=train['sqft_living'].values, y=np.log1p(train['price']), 
             size = 10, ratio = 7, joint_kws={'line_kws':{'color':'limegreen'}},
              kind='reg',color="#34495e")
plt.title('Joint Plot Area Vs Price')
plt.ylabel('Price', fontsize=12)
plt.xlabel('Living Sqft', fontsize=12)
plt.show()


# In[ ]:


sns.set()
cols = ['price', 'sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bedrooms','bathrooms']
sns.pairplot(train[cols], size = 2.5,  palette='afmhot')
plt.show();


# In[ ]:


from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV 
from sklearn.metrics import r2_score
import lightgbm as lgb


# In[ ]:


Xtrain = train.drop('price',axis=1)
ytrain = np.log1p(train.price.values)


# In[ ]:


Xtrain.head()


# In[ ]:


def lgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds), 'name'

X_tr, X_te, y_tr, y_te = train_test_split(Xtrain, ytrain, test_size=0.2, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X_tr, y_tr, test_size=0.2, random_state=4)
lgb_params = {}
lgb_params['boost'] = 'gbdt'
lgb_params['objective'] = 'regression_l2'
lgb_params['num_leaves'] = 128
lgb_params['sub_feature'] = 0.8 
lgb_params['max_depth'] = 9
lgb_params['feature_fraction'] = 0.7
lgb_params['bagging_fraction'] = 0.7
lgb_params['bagging_freq'] = 50
lgb_params['learning_rate'] = 0.01
lgb_params['num_iterations'] = 1500
lgb_params['early_stopping_round'] = 50
lgb_params['verbose'] = 2


ytra = y_train.ravel()
yte = y_test.ravel()
lgb_train = lgb.Dataset(X_train, label=ytra)
lgb_test = lgb.Dataset(X_test, label=yte)
lightgbm = lgb.train(lgb_params, lgb_train, num_boost_round=1500, verbose_eval=100, feval = lgb_r2_score,
                     valid_sets=[lgb_train,lgb_test])
print('LGB Model R2 Score: ', r2_score(np.expm1(lightgbm.predict(X_te)), np.expm1(y_te)))


# # R 2 square : 0.832

# In[ ]:


print('Plot feature importances...')
plt.figure(figsize=(12,8))
lgb.plot_importance(lightgbm)
plt.show()


# # Stay tune
