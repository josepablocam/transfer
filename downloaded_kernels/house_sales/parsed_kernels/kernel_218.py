#!/usr/bin/env python
# coding: utf-8

#  House price prediction using regression
# 
# *********************************************************************
# 
# The steps followed to predict house prices are 
# 
# 1. Data Ingestion 
# 
# 2. Data Exploration 
# 
# 3. Data Transformation 
# 
# 4. Feature Selection
# 
# 5. Test train split 
# 
# 6. Prediction

# In[ ]:


import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Data Ingestion using pandas
contents = pd.read_csv('../input/kc_house_data.csv')


# **Data Exploration using Pandas**

# In[ ]:


# Data Exploration 
contents.head()


# In[ ]:


# Features 
len(contents.columns)


# *Understanding continuous and categorical attributes*

# In[ ]:


contents.info()


# In[ ]:


contents.get_dtype_counts()


# **Data Exploration**
# 
# Here we perform,
# 
# Univariate Analysis: analyzing individual features.
# 
# Bi Variate analysis: analyzing features together.

# ***Univariate Analysis***
# --------------------------------------------------
# 
# For Continuous variables,we can find mean, median and IQR. 
# 
# Histograms, BoxPlot and Violin plots are used for visualization

# In[ ]:


# data for mean house price 
contents.describe()


# In[ ]:


# get the mean price for the house 
target = contents['price'].tolist()
mean_price = sum(target)/len(target)
print(mean_price)


# In[ ]:


# Data for the mean,high and low sales price 
meanrange = contents[(contents.price > 540000) & (contents.price <= 550000) ]
lowrange = contents[(contents.price > 70000) & (contents.price <= 75000) ]
highrange = contents[(contents.price > 7000000) & (contents.price <= 7700000 ) ]


# In[ ]:


low_price = min(target)
print(low_price)
high_price = max(target)
print(high_price)


# In[ ]:


print("Out of 21613 records")
print(("The records in mean range", len(meanrange)))
print(("The records in high range", len(highrange)))
print(("The records in low range", len(lowrange)))


# In[ ]:


len(contents)


# In[ ]:


low_price = min(target)
print(low_price)


# In[ ]:


#Bar Plots for 'Bedroom' feature in the given dataset
contents.bedrooms.value_counts().plot(kind = 'bar')


# In[ ]:


contents.boxplot(['lat'])


# In[ ]:


contents.boxplot(['long'])


# In[ ]:


contents.boxplot([ 'sqft_lot', 'sqft_living'])


# From the above box plot visualizations, we understand that the outlier removal should be performed in the data cleaning process
# 
# ------------------------------------------------------------------
# 
# **Violin Plots**

# In[ ]:


import seaborn as sns
sns.set(color_codes=True)


# In[ ]:


sns.violinplot(contents['yr_renovated'], color = 'cyan')


# In[ ]:


sns.violinplot(contents['yr_built'], color = 'cyan')


# **Skewness and Kurtosis analysis**

# In[ ]:


from scipy import stats
stats.skew(contents.sqft_living, bias=False)


# In[ ]:


stats.skew(contents.sqft_lot15, bias = False)


# In[ ]:


stats.kurtosis(contents.sqft_living15, bias=False)


# In[ ]:


stats.kurtosis(contents.sqft_lot15, bias=False)


# **BiVariate Analysis**
# -----------------------------------------------------
# 
# Here we use scatterplots for our analysis

# ****Linear Correlation between features** **

# In[ ]:


lin_cor = contents.corr(method = 'pearson')['price']
lin_cor = lin_cor.sort_values(ascending=False)
print(lin_cor)


# **Visualization of linear correlation**

# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(target,contents.sqft_living)


# In[ ]:


plt.scatter(target,contents.sqft_lot15)


# In[ ]:


plt.scatter(target,contents.yr_renovated)


# In[ ]:


plt.scatter(target,contents.grade)


# In[ ]:


plt.scatter(target, contents.long)


# In[ ]:


plt.scatter(target, contents.zipcode)


#  Data Cleaning and transformation 
# -------------------------------------------------
# This stage handles,
# 
# 1. removal of missing values 
# 
# 2. Data Feature transformation: Extract the year attribute and encode the year
# 
# 3. Removal of the column id as it has no impact on the price
# 
# 4. Zscore to remove outliers

# In[ ]:


contents.isnull().values.any()


# In[ ]:


# Convert date to year 
date_posted = pd.DatetimeIndex(contents['date']).year


# In[ ]:


conv_dates = [1 if values == 2014 else 0 for values in date_posted ]
contents['date'] = conv_dates


# In[ ]:


contents.date.value_counts().plot(kind = 'bar')


# In[ ]:


contents = contents.drop('id', axis = 1)


# In[ ]:


contents.describe()


# **Removing outliers **

# In[ ]:


import numpy as np
from scipy import stats
contents= contents[(np.abs(stats.zscore(contents)) < 3).all(axis=1)]


# In[ ]:


contents.boxplot([ 'sqft_lot', 'sqft_living'])


# In[ ]:


contents.boxplot(['long'])


#  Feature Selection
# ---------------------------------------------
# 
# For dimensionality reduction, 
# we have used
# 
# 1. PCA 
# 2. Stability Selection

# In[ ]:


predictors = contents.drop('price', axis = 1)
price = contents['price'].tolist()


# **1. Using PCA**

# In[ ]:


#Standardize the data to input to PCA
from sklearn.preprocessing import scale
std_inputs = scale(predictors)
res_inputs = std_inputs.reshape((-1,19))
std_df = pd.DataFrame(data=std_inputs,columns= predictors.columns)


# In[ ]:


# 1. Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
pca = PCA()   
pca = PCA().fit_transform(std_inputs)


# **Kaisers criterion**

# In[ ]:


a = list(np.std((pca), axis=0))
summary = pd.DataFrame([a])
summary = summary.transpose()
summary.columns = ['sdev']
summary.index = predictors.columns
kaiser = summary.sdev ** 2
print(kaiser)


# **Scree Plot**

# In[ ]:


y = np.std(pca, axis=0)**2
x = np.arange(len(y)) + 1
plt.plot(x, y, "o-")
plt.show()


# **2.Stability selection**

# In[ ]:


import time 
from sklearn.linear_model import RandomizedLasso
rlasso = RandomizedLasso(alpha=0.025)


# In[ ]:


get_ipython().run_line_magic('time', 'rlasso.fit(predictors, price)')


# In[ ]:


names = predictors.columns
print((sorted(zip([round(x, 4) for x in rlasso.scores_], 
                 names), reverse=True)))


# In[ ]:


final_predictors = predictors.drop(['yr_renovated', 'waterfront'], axis = 1)


# Train test split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_predictors, price, test_size=0.33, random_state=42)


#  Prediction
# ----------------------------------------
# The regression algorithms used
# 1. Linear Regression
# 2. Gradient Boosting machine (GBM)

# In[ ]:


#Linear regression 
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)


# In[ ]:


#r2 score 
regr.score(X_test,y_test)


# In[ ]:


#GBM model
from sklearn import ensemble
params = {'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 2,
          'learning_rate': 0.1, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


#r^2 score
clf.score(X_test,y_test)


#  **Log loss**

# In[ ]:


test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')


# **Variable Importances**

# In[ ]:


feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, final_predictors.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

