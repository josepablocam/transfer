#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
import math
from __future__ import division
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation, tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score


# # 1. Exploratory Data Analysis

# In[ ]:


# Read the data into a data frame
data = pd.read_csv('../input/kc_house_data.csv')


# In[ ]:


# Check the number of data points in the data set
print(len(data))
# Check the number of features in the data set
print(len(data.columns))
# Check the data types
print(data.dtypes.unique())


# - Since there are Python objects in the data set, we may have some categorical features. Let's check them. 

# In[ ]:


data.select_dtypes(include=['O']).columns.tolist()


# - We only have the date column which is a timestamp that we will ignore.

# In[ ]:


# Check any number of columns with NaN
print(data.isnull().any().sum(), ' / ', len(data.columns))
# Check any number of data points with NaN
print(data.isnull().any(axis=1).sum(), ' / ', len(data))


# - The data set is pretty much structured and doesn't have any NaN values. So we can jump into finding correlations between the features and the target variable

# # 2. Correlations between features and target

# In[ ]:


features = data.iloc[:,3:].columns.tolist()
target = data.iloc[:,2].name


# In[ ]:


correlations = {}
for f in features:
    data_temp = data[[f,target]]
    x1 = data_temp[f].values
    x2 = data_temp[target].values
    key = f + ' vs ' + target
    correlations[key] = pearsonr(x1,x2)[0]


# In[ ]:


data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]


# - We can see that the top 5 features are the most correlated features with the target "price"
# - Let's plot the best 2 regressors jointly

# In[ ]:


y = data.loc[:,['sqft_living','grade',target]].sort_values(target, ascending=True).values
x = np.arange(y.shape[0])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.subplot(3,1,1)
plt.plot(x,y[:,0])
plt.title('Sqft and Grade vs Price')
plt.ylabel('Sqft')

plt.subplot(3,1,2)
plt.plot(x,y[:,1])
plt.ylabel('Grade')

plt.subplot(3,1,3)
plt.plot(x,y[:,2],'r')
plt.ylabel("Price")

plt.show()


# # 3. Predicting House Sales Prices

# In[ ]:


# Train a simple linear regression model
regr = linear_model.LinearRegression()
new_data = data[['sqft_living','grade', 'sqft_above', 'sqft_living15','bathrooms','view','sqft_basement','lat','waterfront','yr_built','bedrooms']]


# In[ ]:


X = new_data.values
y = data.price.values


# In[ ]:


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y ,test_size=0.2)


# In[ ]:


regr.fit(X_train, y_train)
print(regr.predict(X_test))


# In[ ]:


regr.score(X_test,y_test)


# - Prediction score is about 70 which is not really optimal

# In[ ]:


# Calculate the Root Mean Squared Error
print("RMSE: %.2f"
      % math.sqrt(np.mean((regr.predict(X_test) - y_test) ** 2)))


# In[ ]:


# Let's try XGboost algorithm to see if we can get better results
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)


# In[ ]:


traindf, testdf = train_test_split(X_train, test_size = 0.3)
xgb.fit(X_train,y_train)


# In[ ]:


predictions = xgb.predict(X_test)
print(explained_variance_score(predictions,y_test))


# - Our accuracy is changing between 79%-84%. I think it is close to an optimal solution.

# In[ ]:




