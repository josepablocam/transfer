#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# I will begin with some data exploration

# In[ ]:


train = pd.read_csv('../input/kc_house_data.csv')


# In[ ]:


train.info()


# Except date, all features are numeric. Let's have some insight on data.

# In[ ]:


train.head()


# From first sight, data seems OK. Let's dive deeper to understand it.'

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
train.hist(bins=30, figsize=(14,12))
plt.show()


# Notes:
# * There is some sparse data. So we have to look for possible outliers
# * Create dummy variables for sqft_basement and yr_renovated because 0 value means that value is absent

# In[ ]:


train.describe()


# Let's now look for correlations with price feature

# In[ ]:


correlations = train.corr()


# In[ ]:



def heatmap_gen(corr):
    sns.set(style='white')
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    plt.figure(figsize=(9,8))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr * 100, annot=True, fmt='.0f', mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    


# In[ ]:


heatmap_gen(correlations)


# It is clear that there is some linear correlation between price and (grade and sqft_living)

# In[ ]:


sns.pairplot(train[['sqft_living', 'grade', 'bathrooms', 'price']])
plt.show()


# In[ ]:


sns.boxplot(x='grade', y='price', data=train)
plt.show()


# In[ ]:


train.plot.scatter(x='grade', y='price', s=train.sqft_living * 0.05,alpha=0.2, c='waterfront',colormap='plasma',edgecolors='grey', figsize=(10,10))
plt.show()


# According to last graph, we can roughly say that price increases with increasing grade and sqft_living. Also, for the same grade and size, houses in front of water seems to have higher prices. 

# Now let's verify data consistency

# In[ ]:


train.isnull().any()


# In[ ]:


for feature in ['bedrooms', 'bathrooms', 'floors', 'view', 'condition', 'grade', 'yr_built', 'yr_renovated', 'waterfront']:
    print(feature, train[feature].sort_values().unique())


# I'm going to verify some suspecious values like bedrooms=33 or bathrooms=0

# In[ ]:


train[train.bathrooms == 0]


# It looks like it is not an outlier and I'm not sure if this is a mistake so I'm going to leave these rows

# In[ ]:


sns.violinplot(x='bedrooms', data=train)
plt.show()


# In[ ]:


train[train.bedrooms > 30]


# I'm going to delete this row because I think it is a big outlier

# In[ ]:


train_cp = train.copy()
train_cp = train_cp[train_cp.bedrooms < 30]


# Now I'm going to verify data consistency. For example, is yr_renovated always bigger than yr_built

# In[ ]:


train_cp['diff_renov_built'] = train_cp.yr_renovated - train_cp.yr_built


# In[ ]:


train_cp[(train_cp.diff_renov_built < 0) & (train_cp.yr_renovated !=0)]


# Now let's create dummy variables for renovated and basement

# In[ ]:


train_cp['isrenovated'] = (train_cp.yr_renovated != 0).astype(int)
train_cp['hasbasement'] = (train_cp.sqft_basement != 0).astype(int)


# Now let's procede to feature engineering.  I will extract year sold from date feature

# In[ ]:


train_cp['yr_sold'] = train_cp['date'].str[:4].astype(int)


# In[ ]:


train_cp['age'] = train_cp.yr_sold - train_cp.yr_built
train_cp['yr_after_renov'] = train_cp.yr_sold - train_cp.yr_renovated - train_cp.yr_built * (1 - train_cp.isrenovated)


# Now I want to see if there are some clusters in house structures.

# In[ ]:


train_cp.plot.scatter(x='bedrooms', y='bathrooms', alpha=0.1, s=train_cp.price*0.0001, figsize=(10,10))
plt.show()


# It is not clear if there is interessant clusters or if it is possible to extract some. If you have some idea It would be nice of you to share it with me :)

# In[ ]:


train_cp.plot.scatter(x='long', y='lat', c='price', colormap='jet', alpha=0.3, edgecolor='grey', figsize=(10,10))
plt.show()


# From this graph, I see that area within lat > 47.5 contains more expensive houses. So, I'm going to create a feature to distinguish this area

# In[ ]:


train_cp['lat>47.5'] = (train_cp.lat >= 47.5).astype(int)


# In[ ]:


train_cp.head()


# Now I'm going to drop non relevant features

# In[ ]:


train_cp = train_cp.drop(['date', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'yr_sold'], axis=1)


# In[ ]:


train_cp = train_cp.set_index('id')


# Now it's time to train some models.

# In[ ]:


from sklearn.model_selection import train_test_split


# Now I'm going to split data and prepare pipelines

# In[ ]:


y = train_cp.price
X = train_cp.drop('price', axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
pipelines = {
    'elsnet' : make_pipeline(StandardScaler(), ElasticNet(random_state=123)),
    'rf' : make_pipeline(StandardScaler(), RandomForestRegressor(random_state=123)),
    'gb' : make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=123)),
}


# Prepare hyperparameters

# In[ ]:


elsnet_hyper = {
'elasticnet__alpha': [0.05, 0.1, 0.5, 1, 5, 10],
'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]
}

rf_hyper = {
'randomforestregressor__n_estimators' : [100, 200],
'randomforestregressor__max_features': ['auto', 'sqrt', 0.33],
}

gb_hyper = {
'gradientboostingregressor__n_estimators': [100, 200],
'gradientboostingregressor__learning_rate' : [0.05, 0.1, 0.2],
'gradientboostingregressor__max_depth': [1, 3, 5]
}

hyper={
    'elsnet': elsnet_hyper,
    'rf': rf_hyper,
    'gb': gb_hyper
}


# In[ ]:


from sklearn.model_selection import GridSearchCV
fitted_models = {}
for name , pipeline in pipelines.items():
    model = GridSearchCV(pipeline , hyper[name], cv=10)
    model.fit(X_train , y_train)
    fitted_models[name] = model
    print(name)


# In[ ]:


for name, model in fitted_models.items():
    print(name, model.best_score_)


# In[ ]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# In[ ]:


for name , model in fitted_models.items():
    pred = model.predict(X_test)
    print( name )
    print( '------------' )
    print( 'R^2:', r2_score(y_test , pred ))
    print( 'MAE:', mean_absolute_error(y_test , pred))
    print()


# In[ ]:


fitted_models['rf'].best_estimator_


# Based on R2 and mean absolute error I think that best model is random frorest regressor.

# In[ ]:




