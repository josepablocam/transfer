#!/usr/bin/env python
# coding: utf-8

# ## Thank you for visiting my Karnel !

# I have just started with this dataset that impiles House Sales in King County, USA.  My Karnel will be sometime updated by learning from many excellent analysts. 
# 
# * I am not native in English, so very sorry to let you read poor one.

# ## 1.Read libraries and the dataset
# Read libraries and the dataset before analysing.Especially we should care about strange points of the dataset.
# 
#  ## 2.Data Cleaning and Visualizations
# I need to conduct nulls and duplications including strange points above. We also see the relation between 'price' as the target and other valuables from visualizations. We try to evaluate 1st model before feature engineering because of seeing the progress. Then, as explanatory variables increase through feature engineering, multicollinearities are detected.
# 
# * 2-1.Exploring nulls and duplications into the dataset.
# * 2-2.Visualizing the price
# * 2-3.Model building(1st)
# * 2-4-1. Feature engineering: "date"
# * 2-4-2. Feature engineering: "renovation"
# * 2-4-3. Feature engineering: "zipcode"
# * 2-4-4. New dataset
# * 2-4-5. Detecing multicollinearity
# 
# ## 3.Model building and Evaluation
# The model will be built by using train dataset after detecting multicollinearity.  In addition, it is evaluated on the correlation^2 between predicted values (y_pred) and actual values(y_test), MSE(mean_squared_error) and MAE(mean_squared_error)

# ## 1.Read libraries and the dataset

# Anaylsis will be started by reading librariese and the datasets.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ## 1-1. Load the dataset

# In[ ]:


df = pd.read_csv("../input/kc_house_data.csv")
df.head()


# In[ ]:


df.tail()


# ****Dataset shows that the target is 'price' and the other explanatory variables are 20.

# In[ ]:


print(df.shape)
print('------------------------')
print(df.nunique())
print('------------------------')
print(df.dtypes)


# ![](http://)Dataset's shape implies 21,613 lines * 21 columns where are composes as be said above.
# #### It is found that the number of lines(21,613) and id(21,436) is different by 176 except the 1st column of explanatory valuables. It should be caused by some nulls or/and duplications.

# ## 2.Data Cleaning and Visualisation

#  ### 2-1.Exploring nulls and duplications into the dataset.

# In[ ]:


df.isnull().sum()


# In[ ]:


df['id'].value_counts()


# In[ ]:


sum((df['id'].value_counts()>=2)*1)


# It becomes cleared that the difference is cased by **DUPLICATION**, NOT nulls.
# * Also, on other variables, there are NOT nulls which we have to care.
# When my goal is set to predict 'price', show the distribution and fundamental statistics of 'price' and the correlation between 'price' and other valuables except 'id'.

# ### 2-2. Visualizing the price

# Firstly seeing the distribution of price. It may not be directly useful for prediction, however, the clarification of target data is important.

# In[ ]:


plt.hist(df['price'],bins=100)


# In[ ]:


# Seeing the fundamental statistics of price.
df.describe()['price']


# ![](http://)Distribution of price is distorted to the right. The large difference between minimum and maximum price. More than 100 times!!
# * Nextly, seeing the correlation matrix and the scatter plots between "price" and other variables except 'date'.
# * **'date' is needed to change significantly.**

# In[ ]:


df.corr().style.background_gradient().format('{:.2f}')


# In[ ]:


for i in df.columns:
    if (i != 'price') & (i != 'date'):
        df[[i,'price']].plot(kind='scatter',x=i,y='price')


# Though the dtypes of 'yr_renovated' and 'zipcode' are int64, they might be needed to be feature engineered because 'yr_renovated' is focused on around 0 and 2000 from seeing scatter plots above and 'zipcode' is just number.

# ### 2-3. Model Building (1st)

# * Try to biuild 1st model, that the target is 'price' and X are other valuables except 'id',  'date', 'yr_renovated' and 'zipcode'.

# In[ ]:


from sklearn.linear_model import LinearRegression
X = df.drop(['price','id','date','yr_renovated','zipcode'],axis=1)
y = df['price']
regr = LinearRegression(fit_intercept=True).fit(X,y)
print("model_1_score:{:.4f}".format(regr.score(X,y)))


# ### 2-4-1. Feature engineering: "date"

# Firstly, as be said , 'date' will be feature engineered to be significant because 'price' may be related with day of week ('dow') and month.

# In[ ]:


df.date.head()


# In[ ]:


pd.to_datetime(df.date).map(lambda x:'dow'+str(x.weekday())).head()


# ** dow：day of week, 0=Monday, 7=Sunday

# In[ ]:


pd.to_datetime(df.date).map(lambda x:'month'+str(x.month)).head()


# ** month1=January, 12=December

# In[ ]:


df['dow'] = pd.to_datetime(df.date).map(lambda x:'dow'+str(x.weekday()))
df['month'] = pd.to_datetime(df.date).map(lambda x:'month'+str(x.month))


# > Nextly, as the values of 'dow' and 'month' are categorilized, they are changed to be one hot encoding.

# In[ ]:


pd.get_dummies(df['dow']).head()


# In[ ]:


pd.get_dummies(df['month']).head()


# * **The month is not correctly sorted, however the way to revise is not understood to me.

# ### 2-4-2. Feature engineering: "renovation"

# The value of 'yr_renovated'is difficult to be used by itself, therefore, it will be transformed whether the house was renovated or not.

# In[ ]:


df.yr_renovated.head()


# In[ ]:


df['yr_renovated'].value_counts().sort_index().head()


# In[ ]:


np.array(df['yr_renovated'] !=0)


# In[ ]:


np.array(df['yr_renovated'] !=0)*1


# In[ ]:


df['yr_renovated_bin'] = np.array(df['yr_renovated'] != 0)*1
df['yr_renovated_bin'].value_counts()


# ### 2-4-3. Feature engineering: "zipcode"

# The value of zipcode itself may be not directly related with price because it is just the number as below. However, the areas seem to be important valuables for housing price, so it should be changed to be one hot encoding.

# In[ ]:


df['zipcode'].astype(str).map(lambda x:x).head()


# In[ ]:


df['zipcode_str'] = df['zipcode'].astype(str).map(lambda x:'zip_'+x)
pd.get_dummies(df['zipcode_str']).head()


# ### 2-4-4. New dataset

# One hot encoding of 'dow', 'month' and 'zipcode'

# In[ ]:


df['zipcode_str'] = df['zipcode'].astype(str).map(lambda x:'zip_'+x)
df_en = pd.concat([df,pd.get_dummies(df['zipcode_str'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df.dow)],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df.month)],axis=1)


# Dropping the original valuables because feature engineering were conducted.

# In[ ]:


df_en_fin = df_en.drop(['date','zipcode','yr_renovated','month','dow','zipcode_str',],axis=1)


# In[ ]:


print(df_en_fin.shape)
print('------------------------')
print(df_en_fin.nunique())


# In[ ]:


df_en_fin.head()


# ### 2-4-5. Detecing multicollinearity

# Seeing whether the multicollinearity occurs by using these valuables.

# In[ ]:


X = df_en_fin.drop(['price'],axis=1)
y = df_en_fin['price']
regr = LinearRegression(fit_intercept=True).fit(X,y)
model_2 = regr.score(X,y)
for i, coef in enumerate(regr.coef_):
    print(X.columns[i],':',coef)


# ****When seeing the result of regr.coef_, for example, 'bedrooms' is negative against 'price'. Normally 'bedrooms' could be positively proportional with 'price'. However it is caused by strong positive correlation by 0.58 with 'sqft_living'. Because multicollinearity is thought to be occurred in other valuables.
# * **In the case of multicollinearity, VIF value should be considered.

# In[ ]:


df_vif = df_en_fin.drop(["price"],axis=1)
for cname in df_vif.columns:  
    y=df_vif[cname]
    X=df_vif.drop(cname, axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    rsquared = regr.score(X,y)
    print(cname,":" ,1/(1-np.power(rsquared,2)))


# The criteria of multicollinearity is generally over VIF(Variance Inflation Factor) value by 10 or some inf (rsquare==1) are found. Therefore, we derive the valuables to meet criteria of 'rsquare>1.-1e-10', in addition, empirically 'regr.coef_'> |0.5| .

# In[ ]:


df_vif = df_en_fin.drop(["price"],axis=1)
for cname in df_vif.columns:  
    y=df_vif[cname]
    X=df_vif.drop(cname, axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    rsquared = regr.score(X,y)
    #print(cname,":" ,1/(1-np.power(rsquared,2)))
    if rsquared > 1. -1e-10:
        print(cname,X.columns[(regr.coef_> 0.5) | (regr.coef_ < -0.5)])


# Dropping 'sqft_above','zip_98001', 'month1' and 'dow1'.

# In[ ]:


df_en_fin = df_en_fin.drop(['sqft_above','zip_98001','month1','dow1'],axis=1)

df_vif = df_en_fin.drop(["price"],axis=1)
for cname in df_vif.columns:  
    y=df_vif[cname]
    X=df_vif.drop(cname, axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    rsquared = regr.score(X,y)
    #print(cname,":" ,1/(1-np.power(rsquared,2)))
    if rsquared > 1. -1e-10:
        print(cname,X.columns[(regr.coef_> 0.5) | (regr.coef_ < -0.5)])


# NO multicollinearity happens!!

# ## 3.Model building and Evaluation

# The model will be built by using train dataset after detecting multicollinearity.  In addition, it is evaluated on the correlation between predected values (y_pred) and actual values(y_test), MSE(mean_squared_error) and MAE(mean_squared_error)

# In[1]:


X_multi = df_en_fin.drop(['price'],axis=1)
y_target = df_en_fin['price']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X_multi,y_target,random_state=42)


# In[ ]:


regr_train=LinearRegression().fit(X_train,y_train)
y_pred = regr_train.predict(X_test)


# In[ ]:


print("correlation:{:.4f}".format(np.corrcoef(y_test,y_pred)[0,1]))


# In[ ]:


#MAE = mean_absolute_error(y_test,y_pred)
#MSE = mean_squared_error(y_test,y_pred)

print("MAE:{:.2f}".format(mean_absolute_error(y_test,y_pred)),'/ '
      "MSE:{:.2f}".format(mean_squared_error(y_test,y_pred)))


# ## Conclusion 
# 
# 
# 
# ## Next Issues
# 1. Exactly the accuracy(correlation^2) was over 0.8, however, LinearRegression was only tried at this time. Therefore the other methodology should be tried on the purpose to get better.
# 
# 2.  As there are over 100 of the explanatory variables, overfitting may happen.Therefore the number of variables may need to decrease.

# In[ ]:




