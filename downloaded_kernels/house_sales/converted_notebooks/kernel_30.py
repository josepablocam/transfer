#!/usr/bin/env python
# coding: utf-8

# # Feature engineering for a score  .99 using CatBoost and geohash clus
# ## This is achieved with the following assumptions 
# 1. Price is always predicted for future date. Thus train with 80% is data from available dates and we are trying to predict for the future dates
# 2. We calculate price per squarefeet for given location for train data. Additional features derived also will be calculated in train data and copied to test data. This prevents data leaks from test to train. This in general is a known fact when anyone tries to buy a property. 
# 
# Feature ranking is adapted from Anisotropic (https://www.kaggle.com/arthurtok/feature-ranking-rfe-random-forest-linear-models)

# In[ ]:


import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
#import geohash
from catboost import CatBoostRegressor
import catboost


# # Feature engineering helpers 

# In[ ]:


def returnYear(row):
    if row['yr_renovated']!=0:
        return datetime.strptime(str(row['yr_renovated']),'%Y')
    else:
        return row['yr_built']
def deltaInYearsAge(row):
    difference = relativedelta(row['date'], row['yr_built'])
    years = difference.years
    return years
def deltaInYearsRenovated(row):
    difference = relativedelta(row['yr_renovated'], row['yr_built'])
    years = difference.years
    return years


# # Since kaggle does not support geohash libraries, using one from git
# Original source https://github.com/vinsci/geohash/blob/master/Geohash/geohash.py
# The libraies gave a much better result of 0.96 

# In[ ]:


from math import log10
__base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
def geohashEncode(latitude, longitude, precision=12):
    """
    Encode a position given in float arguments latitude, longitude to
    a geohash which will have the character count precision.
    """
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    geohash = []
    bits = [ 16, 8, 4, 2, 1 ]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash += __base32[ch]
            bit = 0
            ch = 0
    return ''.join(geohash)


# # Load Data and define the target variable

# In[ ]:


house = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
print (house.shape)
house.drop_duplicates('id',inplace=True)
print(house.shape)
targetVariableColumnName = 'price'


# In[ ]:


house.columns


# # creating features based on location. Geohash with different accuracies is handy for clustering/grouping

# In[ ]:


house['date'] = pd.to_datetime(house['date'])
house.sort_values('date',inplace=True)
house['yr_built'] = house.yr_built.apply(lambda x:datetime.strptime(str(x),'%Y') )
house['yr_renovated'] = house.apply(returnYear,axis=1)
house['age']=house.apply(deltaInYearsAge,axis=1)
house['renovatedAge']=house.apply(deltaInYearsRenovated,axis=1)
house['geohash']=house.apply(lambda points: geohashEncode(points.lat, points.long,precision=4),axis = 1)
house['pricepersqft']=house['price']/house['sqft_living']


# In[ ]:


house.shape[0]*0.8


# In[ ]:


train = house.head(17148)


# # Groupby functions on getting bias over neighborhood

# In[ ]:


train=train.join(train.groupby(['geohash'])['pricepersqft'].mean(),on='geohash',rsuffix='priceaverage600m')
train=train.join(train.groupby(['geohash'])['pricepersqft'].min(),on='geohash',rsuffix='pricemin600m')
train=train.join(train.groupby(['geohash'])['pricepersqft'].max(),on='geohash',rsuffix='pricemax600m')

train=train.join(train.groupby(['geohash'])['pricepersqft'].max(),on='geohash',rsuffix='pricemax600m')


# In[ ]:


pd.set_option('display.float_format', lambda x: '%.0f' % x)
print (train.shape)
train.drop_duplicates('id',inplace=True)
print (train.shape)

train.describe().T


# In[ ]:


test = house.tail(4465)
test.to_csv('original_test.csv')

currentIds=set(test['id'].values)
print (test.shape)
test=pd.merge(test, train[['geohash','pricepersqftpriceaverage600m','pricepersqftpricemin600m', 'pricepersqftpricemax600m']], on="geohash")
test.drop_duplicates('id',inplace=True)
test.to_csv('merged_test.csv')
currentIds1=set(test['id'].values)
print (currentIds.difference(currentIds1))
print (test.shape)


# # now drop the items already covered in added features
# zip code, lat, lon are covered in addl features with respect to location 
# year renowated and built are added as age and renowated age
# other columns logprice, geohash ... 

# In[ ]:


columns=list(train.columns.values)
columns.remove(targetVariableColumnName)

columns=[item for item in columns if item not in ['zipcode', 'lat','long','id','yr_renovated','yr_built','date','geohash','geohash_70m','Log_price']]
print (columns)


# # Feature ranking 
# 

# In[ ]:


# First extract the target variable which is our House prices
Y = train.price.values
# Drop price from the house dataframe and create a matrix out of the house data
X = train[columns].as_matrix()
# Store the column/feature names into a list "colnames"
colnames = columns


# In[ ]:


ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


# In[ ]:


from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
rf.fit(X,Y)
ranks["RF"] = ranking(rf.feature_importances_, colnames);


# In[ ]:


# Finally let's run our Selection Stability method with Randomized Lasso
rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X, Y)
ranks["rlasso/Stability"] = ranking(np.abs(rlasso.scores_), colnames)
print('finished')


# In[ ]:


# Construct our Linear Regression model
lr = LinearRegression(normalize=True)
lr.fit(X,Y)
#stop the search when only the last feature is left
rfe = RFE(lr, n_features_to_select=1, verbose =3 )
rfe.fit(X,Y)
ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)


# In[ ]:


# Using Linear Regression
lr = LinearRegression(normalize=True)
lr.fit(X,Y)
ranks["LinReg"] = ranking(np.abs(lr.coef_), colnames)

# Using Ridge 
ridge = Ridge(alpha = 7)
ridge.fit(X,Y)
ranks['Ridge'] = ranking(np.abs(ridge.coef_), colnames)

# Using Lasso
lasso = Lasso(alpha=.05)
lasso.fit(X, Y)
ranks["Lasso"] = ranking(np.abs(lasso.coef_), colnames)


# In[ ]:


# Create empty dictionary to store the mean value calculated from all the scores
r = {}
for name in colnames:
    r[name] = round(np.mean([ranks[method][name] 
                             for method in ranks.keys()]), 2)
 
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")


# In[ ]:


# Put the mean scores into a Pandas dataframe
meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])

# Sort the dataframe
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)


# In[ ]:


import seaborn as sns

# Let's plot the ranking of the features
sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar", 
               size=11)


# # Let's Predict with CatBoost Library

# In[ ]:


cbc = CatBoostRegressor(random_seed=0).fit(train[columns].values,train[targetVariableColumnName].values)


# In[ ]:


test['predictionsCatBoost'] = cbc.predict(test[columns])


# In[ ]:


from sklearn.metrics import explained_variance_score,median_absolute_error
print (explained_variance_score(test['price'], test['predictionsCatBoost']),median_absolute_error(test['price'], test['predictionsCatBoost']))


# In[ ]:


test['predictionsCatBoost']=test['predictionsCatBoost'].apply(lambda x: int(round(x)))
test[['price','predictionsCatBoost','age','id']].head()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import matplotlib
matplotlib.pyplot.scatter(test['predictionsCatBoost'],test[targetVariableColumnName])


# In[ ]:




