#!/usr/bin/env python
# coding: utf-8

# The house prices depend on various factors and such factors varies across different markets. In this analysis, we use the King County, USA data set to build a regression model that will help predict the house prices in that region given a set of attributes. I will try to identify a set of attributes and use a mixture of regression techniques to see which technique gives the best negative mean squared error.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv('../input/kc_house_data.csv')
data.head(3)


# In[ ]:


print((data.shape))


# Check for missing or null values in the data set. Looks like everything is in place. 

# In[ ]:


print((data.isnull().any()))


# In[ ]:


print((data.dtypes))


# From the output above, some of the columns are with the wrong data types. For example, the data column is of the type object, floors and bathrooms are of the type float64. Also, the id column can be used as the row index and I do suspect the year which the house built will affect the house price.

# ## Some feature engineering
# 
# I will do some data handling here. First, convert the id column to be the index of the data frame. Next, convert the data object to datatime. Finally, convert the data type for price, bathrooms and floors from float to int.

# In[ ]:


data['date'] = pd.to_datetime(data['date'])
data = data.set_index('id')
data.price = data.price.astype(int)
data.bathrooms = data.bathrooms.astype(int)
data.floors = data.floors.astype(int)
data.head(5)


# I will also create a column call house_age that is derived from the subtraction of date and yr_built. I will then drop the yr_built column and data column.
# Next, I create a **renovated** column. If the **yr_renovated** column is a non-zero, I'll set a 1 to the **renovated** column. Then, I will drop the yr_renovated column.

# In[ ]:


data["house_age"] = data["date"].dt.year - data['yr_built']
data['renovated'] = data['yr_renovated'].apply(lambda yr: 0 if yr == 0 else 1)

data=data.drop('date', axis=1)
data=data.drop('yr_renovated', axis=1)
data=data.drop('yr_built', axis=1)
data.head(5)


# In[ ]:


pd.set_option('precision', 2)
print((data.describe()))


# ## Picking out the relevant attributes for regression modelling
# 
# At this point, I pick out the attributes to build the regression modeling. I typically use the method of identifying the top attributes that have direct correlations with the target variable. The target variable is **'price'**. I do this by building a correlation matrix. From the correlation matrix, I pick the top 10 variables that has relationship with the target house price.

# In[ ]:


correlation = data.corr(method='pearson')
columns = correlation.nlargest(10, 'price').index
columns


# In[ ]:


correlation_map = np.corrcoef(data[columns].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlation_map, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns.values, xticklabels=columns.values)

plt.show()


# From the earlier data.describe() output, I observed that the values of 2 variables - price and sqft_living - are large and will affect the absolute numbers of the regression model. To manage this, I will normalise the data using log.

# In[ ]:


data['price'] = np.log(data['price'])
data['sqft_living'] = np.log(data['sqft_living'])


# ## Baseline algorithm test
# 
# There are a few regression algorithms I can use. I prefer to line the usable regression algorithms up and run them using a standard set of data. I check the negative mean square error of each run. The given data set is broken down into training set and testing set. The test set is 20% of the provided data set.

# In[ ]:


X = data[columns]
Y = X['price'].values
X = X.drop('price', axis = 1).values


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.20, random_state=42)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


# The differing scales of the raw data may impact these algorithms. Part of a requirement for a standardised data set is to have each attribute have a mean value of zero and a standard deviation of 1. I implement standardisation using pipelines. I then use cross-validation to validate performance of algorithms in totality.

# In[ ]:


pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=21)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# From the output above, it looks like the Gradient Boosting Regressor operforms the best using a scaled version of the data. From this point onward, I will build the regression algorithm using the Gradient Boosting Regressor. The GBM will be tested with a few n_estimators using the GridSearchCV function.

# In[ ]:


from sklearn.model_selection import GridSearchCV

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=np.array([50,100,200,300,400]))
model = GradientBoostingRegressor(random_state=21)
kfold = KFold(n_splits=10, random_state=21)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(("%f (%f) with: %r" % (mean, stdev, param)))

print(("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)))


# The best n_estimator configuration is 400 with the negative mean square error closest to 0. 
# 
# ## Finalise and validate model 
# 
# There's a need to standardise the training and test data before putting them through the GBR model. 

# In[ ]:


from sklearn.metrics import mean_squared_error

scaler = StandardScaler().fit(X_train)
rescaled_X_train = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=21, n_estimators=400)
model.fit(rescaled_X_train, Y_train)

# transform the validation dataset
rescaled_X_test = scaler.transform(X_test)
predictions = model.predict(rescaled_X_test)
print((mean_squared_error(Y_test, predictions)))


# From the mean square error of 0.046 between the prediction outputs vs the test data, the GBR performed well. Important to note that the mean_square_error is calculated using the scaled data. It does not represent the error between the actual house prices and predicted prices. To better appreciate the outcome of the predictions, I look at the raw predicted values and the corresponding test data.

# In[ ]:


compare = pd.DataFrame({'Prediction': predictions, 'Test Data' : Y_test})
compare.head(10)


# From the data frame output above, the difference between the predicted value and test data is pretty small. Take note that the data is scaled and log normalised. So, we have to inverse transform these data to see the actual values. To do that, I apply the inverse_transform and exp function to the "Prediction" column

# In[ ]:


actual_y_test = np.exp(Y_test)
actual_predicted = np.exp(predictions)
diff = abs(actual_y_test - actual_predicted)

compare_actual = pd.DataFrame({'Test Data': actual_y_test, 'Predicted Price' : actual_predicted, 'Difference' : diff})
compare_actual = compare_actual.astype(int)
compare_actual.head(5)


# Here's the end of my analysis. One thing I would do to improve this is to better handle the feature selection to build the regression model. 
# 
# Any other comments to this is welcomed!
