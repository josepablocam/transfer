#!/usr/bin/env python
# coding: utf-8

# **Author: Jiashen Liu, Data Scientist at [Quantillion](http://quantillion.io/)** 

# # 0. Fire up

# In[ ]:


import pandas as pd
import numpy as np
df = pd.read_csv('../input/kc_house_data.csv')
print(df.shape)


# Let's have a look first on the data we have.

# In[ ]:


df.head()


# Let's drop the id column and split the data set into training and testing sets.

# In[ ]:


del df['id']


# We quickily check: whether there are some missing values exising in our data set before move on.

# In[ ]:


NA_Count = pd.DataFrame({'Sum of NA':df.isnull().sum()}).sort_values(by=['Sum of NA'],ascending=[0])
NA_Count['Percentage'] = NA_Count['Sum of NA']/df.shape[1]


# In[ ]:


sum(NA_Count['Percentage'])


# Data is quite clean! We do not have to waste time on dealing missing values. Let's move on!

# # 1. Exploratory Data Analysis

# First, we split the data set into training and testing sets. EDA will be finished in training set alone.

# In[ ]:


from sklearn.model_selection import train_test_split
train,test = train_test_split(df,test_size = 0.2,random_state=42)


# We can have continous and categorical variables for this case.

# In[ ]:


cat = ['waterfront','view','condition','grade']
con = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','sqft_above','sqft_basement','yr_built','yr_renovated','sqft_living15','sqft_lot15']


# **Lon/Lat vs Price**

# In[ ]:


from ggplot import *
lonlat = ggplot(train,aes(x='long',y='lat',color='price'))+geom_point()+scale_color_gradient(low='white',high='red')+ggtitle('Color Map of Price') 
print(lonlat)


# In[ ]:


lonprice = ggplot(train,aes(x='long',y='price'))+geom_point()+ggtitle('Price VS Longitude')
print(lonprice)


# We do a small feature engineering here. To centralize the longtitude and take absolute values so the new values will be linear with house price. The central point we choose is -122.25.

# In[ ]:


def centralize_long(lon):
    return np.abs(lon+122.25)*-1


# In[ ]:


train['norm_lon'] = train['long'].apply(lambda x: centralize_long(x))
test['norm_lon'] = test['long'].apply(lambda x: centralize_long(x))


# In[ ]:


lonprice2 = ggplot(train,aes(x='norm_lon',y='price'))+geom_point()+ggtitle('Price VS Centered Longitude')
print(lonprice2)


# In[ ]:


latprice = ggplot(train,aes(x='lat',y='price'))+geom_point()+stat_smooth()+ggtitle('Price VS Latitude')
print(latprice)


# **Target VS Postcode**

# In[ ]:


zipprice = ggplot(train,aes(x='zipcode',y='price'))+geom_point()+ggtitle('ZipCode VS Price')
print(zipprice)


# Seems that ZipCode is not a feature that has quite obvious impact on the price. Therefore, it is not quite wise to have such a categorical variable that has so many discrete values. We will try to group those variabes by longtitude and latitude, so we can actually shrink the size of variables. 

# In[ ]:


latlonzip = ggplot(train,aes(x='long',y='lat',color='zipcode'))+geom_point()+ggtitle('Long-Lat VS ZipCode')
print(latlonzip)


# We can see that, zip code in an certain area is continous. So, to simplify everything, we just turn the zipcode into five different areas by specifying their values.

# In[ ]:


def zip2area(zipcode):
    if zipcode <= 98028:
        return 'A'
    elif zipcode>98028 and zipcode <= 98072:
        return 'B'
    elif zipcode>98072 and zipcode<98122:
        return 'C'
    else:
        return 'D'


# In[ ]:


train['Area'] = train['zipcode'].apply(lambda x:zip2area(x))
test['Area'] = test['zipcode'].apply(lambda x:zip2area(x))


# **Target VS Continous Variables**

# In[ ]:


con_train = train[con+['price']]
cor_tar_con = []
for each in con:
    cor_tar_con.append(np.corrcoef(train[each],train['price'])[0][1])
cor_label = pd.DataFrame({'Variables':con,'Correlation':cor_tar_con}).sort_values(by=['Correlation'],ascending=[0])


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib import cm as cm
import seaborn as sns


# In[ ]:


pos_1 = np.arange(len(con))
plt.bar(pos_1, cor_label['Correlation'], align='center', alpha=0.5)
plt.xticks(pos_1, cor_label['Variables'],rotation='vertical')
plt.ylabel('Correlation')
plt.title('Correlation between price and variables') 
plt.show()


# **Multicolinearity among continouse variables**

# In[ ]:


corr = con_train.corr()
corr.style.background_gradient(cmap='viridis', low=.5, high=0).highlight_null('red')


# We can see that: high correlation can be detected between sqft_above and sqft_living. It is suprised that sqft_living and sqft_living15 do not reveal huge correlation in between.

# **Catgorical Variable vs Target**

# In[ ]:


def box_plot(var):
    pt = a = ggplot(train,aes(x=var,y='price'))+geom_boxplot() + theme_bw()+ggtitle('Boxplot of '+var+' and price')
    return print(pt)


# In[ ]:


for each in cat:
    box_plot(each)


# Seems that grade should be treated as a continous variable

# **Price VS Time**

# In[ ]:


train['date']=pd.to_datetime(train['date'])
test['date']=pd.to_datetime(test['date'])


# In[ ]:


dateprice = ggplot(train,aes(x='date',y='price'))+geom_line()+stat_smooth()+ggtitle('Date VS Price')
print(dateprice)


# As the date data is hard to be fitted in the model, we decided to transfer them into the numeric values. We will take the oldest date in the whole data set as the benchmark and calculate the date between the actual date and it.

# In[ ]:


min_date = min(test['date'])
def get_interval(date):
    return int(str(date-min_date).split()[0])


# In[ ]:


train['date_interval'] = train['date'].apply(lambda x: get_interval(x))
test['date_interval']=test['date'].apply(lambda x: get_interval(x))


# **Preparing the data sets**

# In[ ]:


columns = con + cat + ['date_interval','norm_lon','Area']
train_ = train[columns]
test_ = test[columns]
train_['Area']=pd.factorize(train_['Area'], sort=True)[0]
test_['Area']=pd.factorize(test_['Area'], sort=True)[0]


# **Choosing the labels**

# In[ ]:


import statsmodels.api as sm
fig=sm.qqplot(train['price'])
plt.show()


# The QQ Plot reveals a non-normalized distribution of the label. Let's try logrithm transform. 

# In[ ]:


fig=sm.qqplot(np.log(train['price']))
plt.show()


# It is way better. Let's use it as the label in regression Model.

# In[ ]:


train_['log_price'] = np.log(train['price'])


# # 2. Regression Models

# Grid Search maybe the method we use in the process of hyper parameter tuning.

# In[ ]:


Models = []
RMSE = []


# ## 2.1 Linear Regression

# In[ ]:


from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse


# In[ ]:


Models.append('Normal Linear Regression')
reg = LinearRegression(n_jobs=-1)
reg.fit(train_[columns],train_['log_price'])
pred = np.exp(reg.predict(test_))
Accuracy = sqrt(mse(pred,test['price']))
print('=='*20+'RMSE: '+str(Accuracy)+'=='*20)
RMSE.append(Accuracy)


# ## 2.2 Linear Regression With Step 2&3 Polynomial Transformation

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
pipe = Pipeline([
('sc',StandardScaler()),
('poly',PolynomialFeatures(include_bias=True)),
('reg',LinearRegression())
])
model = GridSearchCV(pipe,param_grid={'poly__degree':[2,3]})
model.fit(train_[columns],train_['log_price'])
degree = model.best_params_
print(degree)
pred = np.exp(model.predict(test_))
Accuracy = sqrt(mse(pred,test['price']))
print('=='*20+'RMSE: '+str(Accuracy)+'=='*20)
RMSE.append(Accuracy)


# In[ ]:


Models.append('LinearRegression Step2 Polynominal')


# ## 2.3 Lasso Regression With Step 2 Polynomial Transformation

# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:


pipe = Pipeline([
('sc',StandardScaler()),
('poly',PolynomialFeatures(degree=2,include_bias=True)),
('las',Lasso())
])
model = GridSearchCV(pipe,param_grid={'las__alpha':[0.0005,0.001,0.01]})
model.fit(train_[columns],train_['log_price'])
degree = model.best_params_
print(degree)
pred = np.exp(model.predict(test_))
Accuracy = sqrt(mse(pred,test['price']))
print('=='*20+'RMSE: '+str(Accuracy)+'=='*20)
RMSE.append(Accuracy)
Models.append('Lasso')


# ## 2.4 ElasticNet Regression With Step 2 Polynomial Transformation

# In[ ]:


from sklearn.linear_model import ElasticNet
pipe = Pipeline([
('sc',StandardScaler()),
('poly',PolynomialFeatures(degree=2,include_bias=True)),
('en',ElasticNet())
])
model = GridSearchCV(pipe,param_grid={'en__alpha':[0.005,0.01,0.05,0.1],'en__l1_ratio':[0.1,0.4,0.8]})
model.fit(train_[columns],train_['log_price'])
degree = model.best_params_
print(degree)
pred = np.exp(model.predict(test_))
Accuracy = sqrt(mse(pred,test['price']))
print('=='*20+'RMSE: '+str(Accuracy)+'=='*20)
RMSE.append(Accuracy)
Models.append('ElasticNet Regression')


# ## 2.5 Regression Model Summary

# In short, we can find that, adding features by polynomial transformation can boost the performance of the model to a large extent.

# In[ ]:


RegSummary = pd.DataFrame({'Model':Models,'RMSE':RMSE})
summary = ggplot(RegSummary,aes(x='Model',weight='RMSE'))+geom_bar()+theme_bw()+ggtitle('Summary of Regression Model')
print(summary)


# ## 3. Short Conclusion and TO-DO

# We already tested different regression models. They look OK, but not good enough. Next step, We will put the L2 Regularization as a process of feature selection and put the pipeline into the tree models to see whether a combination can build a state of art model.

# In[ ]:




