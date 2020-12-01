#!/usr/bin/env python
# coding: utf-8

# # My first kaggle competitions.
# [House Sales in King County, USA] KuwaKuwa '02/14/2018

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools 
from sklearn import linear_model 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.linear_model import LinearRegression


# ### Load the dataset.

# Memo.  
# https://www.kaggle.com/harlfoxem/housesalesprediction/data

# In[ ]:


df_data = pd.read_csv('../input/kc_house_data.csv')
df_data["price"] = df_data["price"] / 10**6 
df_data["sqft_living"] = df_data["sqft_living"] / 10**3 
df_data["sqft_living15"] = df_data["sqft_living15"] / 10**3 
df_data["sqft_lot"] = df_data["sqft_lot"] / 10**3 
df_data["sqft_lot15"] = df_data["sqft_lot15"] / 10**3 
df_data["sqft_above"] = df_data["sqft_above"] / 10**3 
df_data["sqft_basement"] = df_data["sqft_basement"] / 10**3 
# df_data = df_data.drop(["id","date","zipcode"], axis=1) 


# In[ ]:


df_data.info()


# Ref.  
# [https://www.kaggle.com/harlfoxem/housesalesprediction/data] Column Metadata  
# id				a notation for a house   
# Data 			Date house was sold   
# price			Price is prediction target <--(*)    
# Bedrooms 		Number of Bedrooms/House   
# bathrooms		Number of bathrooms/bedrooms   
# sqft_living 	square footage of the home   
# sqft_lot 		square footage of the lot   
# floors			Total floors (levels) in house   
# waterfront		House which has a view to a waterfront   
# view 			Has been viewed   
# condition		How good the condition is ( Overall )   
# grade			overall grade given to the housing unit, based on King County grading system   
# sqft_above		square footage of house apart from basement   
# sqft_basement	square footage of the basement   
# yr_built		Built Year   
# yr_renovated	Year when house was renovated   
# zipcode			zip   
# lat				Latitude coordinate   
# long			Longitude coordinate   
# sqft_living15	Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area   
# sqft_lot15		lotSize area in 2015(implies-- some renovations)   

# In[ ]:


display(df_data.shape)
display(df_data.head())
display(df_data.tail())


# In[ ]:


df_data.describe() 


# ### Predict explanatory variables.

# In[ ]:


df_data.corr()


# (*)Ans.  
# 1. sqft_living(0.702035)
# 2. grade(0.667434)
# 3. sqft_above(0.605567)
# 4. sqft_living15(0.585379)
# 5. bathrooms(0.525138)

# ### View some graphs.

# In[ ]:


y_var = "price"
X_var = ["sqft_living","grade","sqft_above","sqft_living15","bathrooms"]
df = df_data[[y_var]+ X_var]

pd.plotting.scatter_matrix(df,alpha=0.3,s=10, figsize=(15,15))
plt.show()


# ### Confirm a strong relationship between 'price' and explanatory variables.

# In[ ]:


import itertools
key=['price']
df_data_trim = df_data[['sqft_living','grade','sqft_above','sqft_living15','bathrooms']]
li_combi = list(itertools.product(key,df_data_trim.columns))
for X,Y in li_combi:
    print(("X=%s"%X,"Y=%s"%Y))
    df_data.plot(kind="scatter",x=X,y=Y,alpha=0.7,s=10,c="price",colormap="winter")
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.tight_layout()
    plt.show()


# ### Confirm missing values.

# In[ ]:


pd.DataFrame(df_data.isnull().sum(), columns=["num of missing"])


# ★Ans.  
# There are nothing of missing values.

# ### Confirm abnormal values.

# In[ ]:


# sqft_living, sqft_living15
df_data.plot(kind="scatter",x="price",y="sqft_living")
df_data.plot(kind="scatter",x="price",y="sqft_living15")
plt.show()


# memo.  
# There is a remotely relationship between 'sqft_living' and 'sqft_living15'.

# In[ ]:


# grade
df_data.boxplot(column="price",by="grade")
plt.title("")
plt.ylabel("price")
plt.show()


# In[ ]:


# sqft_above
df_data.plot(kind="scatter",x="price",y="sqft_above")
plt.show()


# In[ ]:


# bedrooms
df_data.boxplot(column="price",by="bedrooms")
plt.title("")
plt.ylabel("price")
plt.show()


# In[ ]:


# bedrooms
display(df_data['bedrooms'].value_counts())
display(df_data[(df_data['bedrooms']>=10)])


# ★Ans.  
# - 'price' is over 6$MM.
# - 'bedrooms' is over 33.

# ### Calculate evaluation index. (temporarily)

# In[ ]:


# not standardization
y = df_data["price"].values
X = df_data[['sqft_living','grade','sqft_above','sqft_living15','bathrooms']].values

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

kf = KFold(n_splits=5, random_state=1234, shuffle=True)
kf.get_n_splits(X, y)

df_result = pd.DataFrame()

for train_index, test_index in kf.split(X, y):
    print(("TRAIN:", train_index, "TEST:", test_index))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    regr = LinearRegression(fit_intercept=True)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    df = pd.DataFrame({"y_test":y_test, "y_pred":y_pred})
    df_result = pd.concat([df_result, df], axis=0)

y_test = df_result["y_test"]
y_pred = df_result["y_pred"]
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(("MSE=%s"%round(mse,3) ))
print(("RMSE=%s"%round(np.sqrt(mse), 3) ))
print(("MAE=%s"%round(mae,3) ))


# Memo.  
# MSE=0.062  
# RMSE=0.248  
# MAE=0.161  
# 
# 'MSE/RMSE/MAE' are very low values. Why?

# In[ ]:


# standardization
y = df_data["price"].values
X = df_data[['sqft_living','grade','sqft_above','sqft_living15','bathrooms']].values

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()

kf = KFold(n_splits=5, random_state=1234, shuffle=True)
kf.get_n_splits(X, y)

df_result = pd.DataFrame()

for train_index, test_index in kf.split(X, y):
    print(("TRAIN:", train_index, "TEST:", test_index))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

#    y_train_std = stdsc.fit_transform(y_train)
#    y_test_std = stdsc.transform(y_test)

    regr = LinearRegression(fit_intercept=True)
    regr.fit(X_train_std, y_train)
    y_pred = regr.predict(X_test_std)
    df = pd.DataFrame({"y_test":y_test, "y_pred":y_pred})
    df_result = pd.concat([df_result, df], axis=0)

y_test = df_result["y_test"]
y_pred = df_result["y_pred"]
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(("MSE=%s"%round(mse,3) ))
print(("RMSE=%s"%round(np.sqrt(mse), 3) ))
print(("MAE=%s"%round(mae,3) ))


# ### Predict again explanatory variables.

# In[ ]:


# Stepwise
import pyper 
df_data_tmp = df_data.drop(["id","date","zipcode"], axis=1) 
r = pyper.R(use_pandas='True')
r.assign('df_data_tmp', df_data_tmp)
print((r("step(lm(price~.,data=df_data_tmp))")))


# Memo.  
# 1. sqft_living(1.464e-01)
# 2. grade(9.726e-02)
# 3. sqft_above(3.327e-02)
# 4. sqft_living15(2.734e-02)
# 5. bathrooms(4.234e-02)

# In[ ]:


import statsmodels.api as sm 

count = 1
for i in range(5):
    combi = itertools.combinations(df_data.drop(["price","id","date","zipcode"],axis=1).columns, i+1) 
    for v in combi:
        y = df_data["price"]
        X = sm.add_constant(df_data[list(v)])
        model = sm.OLS(y, X).fit()
        if count == 1:
            min_aic = model.aic
            min_var = list(v)
        if min_aic > model.aic:
            min_aic = model.aic
            min_var = list(v)
        count += 1
#        print("AIC:",round(model.aic), "variable:",list(v))
print("====minimam AIC====")
print((min_var,min_aic))


# Memo.  
# ====minimam AIC====  
# ['sqft_living', 'waterfront', 'grade', 'yr_built', 'lat'] -6214.98880309  

# In[ ]:


# Multicollinearity(多重共線性).
y=df_data["price"].values
X=df_data[["sqft_living","grade","sqft_above","sqft_living15","bathrooms"]].values
regr = LinearRegression(fit_intercept=True)
regr.fit(X, y)
print(("coefficient of determination=%s"%regr.score(X,y)))
print(("slope=%s"%regr.coef_,"intercept=%s"%regr.intercept_))


# In[ ]:


# VIF.
df = df_data.drop(["price","id","date","zipcode"],axis=1)
for cname in df.columns:  
    y=df[cname]
    X=df.drop(cname, axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    rsquared = regr.score(X,y)
    print((cname,":" ,1/(1-np.power(rsquared,2))))


# In[ ]:


# L1正則化項(Lasso)をつけて、影響度の低い変数を取り除く


# In[ ]:




