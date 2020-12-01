#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import module 
import numpy as np 
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/kc_house_data.csv')


# In[ ]:


df.shape


# In[ ]:


df.isnull().any() # check any null data 


# In[ ]:


df.describe() # check data scale 


# In[ ]:


df.head() # check data range 


# In[ ]:


# common sense, the id and date should not be related with price. 
# the rest property should be related with price, there are two column data might need some transfrom
# first : check the data distribution of yr_renovated whether too sparse
# second is zip, lat, lon, could be integrated into location. 

# check value of yr_renovated 
df.yr_renovated.value_counts()
price=df.price;yr_renovated=df.yr_renovated
plt.scatter(yr_renovated,price)


# In[ ]:


# lets check the heatmap between each column 
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(),cmap='bone',annot=True,fmt='0.1f',)


# In[ ]:


# so the correlation among price and location(zipcode,lat,lon),sqft_lot15,and condition et al is 
# low , while this might due to the relative scale of this column is too small. 
# so will try use lasso to learning the data, the decide if we want to remove some property 
y=np.array(df['price'])


# In[ ]:


df=df.drop('price',axis=1)
df=df.drop('id',axis=1)
df=df.drop('lat',axis=1)
df=df.drop('long',axis=1)
df=df.drop('date',axis=1)


# In[ ]:


df.head()


# In[ ]:


x=np.array(df)


# In[ ]:


# import lasso 
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn import linear_model


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[ ]:


sc=StandardScaler()
sc.fit(x_train)
x_train_std=sc.transform(x_train)
sc.fit(x_test)
x_test_std=sc.transform(x_test)


# In[ ]:


LM=linear_model.Lasso(alpha=0.1,max_iter=3000)


# In[ ]:


LM.fit(x_train_std,y_train)


# In[ ]:


prediction=LM.predict(x_test_std)
predit=prediction.reshape(-1,1)
y_t=y_test.reshape(-1,1)


# In[ ]:


LM.score(x_test_std,y_t)


# In[ ]:


# 
LSM=linear_model.Ridge(alpha=0.8)


# In[ ]:


LSM.fit(x_train_std,y_train)


# In[ ]:


LSM.score(x_test_std,y_test)


# In[ ]:


# it's too low, lets check if there is any outlier 
import seaborn as sns


# In[ ]:


df['price']=y


# In[ ]:


df.head()


# In[ ]:


x=np.arange(0,21613,1)


# In[ ]:


df[df.bedrooms>9]


# In[ ]:


fig=plt.figure(figsize=(100,100))
#fig.set_figheight(20)
#fig.set_figwidth(20)


# In[ ]:


plt.subplot(1,1,1)
plt.scatter(x,df.iloc[:,1])
#plt.subplot(5,2,2)
#plt.scatter(x,df.iloc[:,2])
#plt.subplot(5,2,3)
#plt.scatter(x,df.iloc[:,3])


# In[ ]:


df[df.bedrooms>9]


# In[ ]:


# records that bedroom is 33 with 1.75 bathroom should be wrong. so change bedroom number into 3 
df.ix[15870,1]=3


# In[ ]:





# In[ ]:


title=df.columns
for i in range(df.shape[1]):
    fig=plt.figure(num=i,figsize=(3,4))
    plt.scatter(x,df.iloc[:,i])
    plt.title(title[i])


# In[ ]:


df[df.bathrooms>6]
# looks ok


# In[ ]:


df[df.sqft_living>10000]
# the lot size for id 1225069038 looks not reasonalbe, its build on 1999, and have large house, but way 
# more cheaper, in case, i will drop this row 
df2=df.drop(12777)


# In[ ]:


df2[df2.sqft_above>8000] # looks fine 
df2[df2.sqft_basement>3000] # also looks fine 


# In[ ]:


# after correct data
#redo the regressions
y=np.array(df2['price'])
x=np.array(df2)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
sc.fit(x_train)
x_train_std=sc.transform(x_train)
sc.fit(x_test)
x_test_std=sc.transform(x_test)


# In[ ]:


y_train=y_train[:,np.newaxis]
y_test=y_test[:,np.newaxis]


# In[ ]:


sc.fit(y_train)
y_train_std=sc.transform(y_train)


# In[ ]:


lm=linear_model.Lasso(alpha=0.5)


# In[ ]:


lm.fit(x_train_std,y_train_std)


# In[ ]:


sc.fit(y_test)
y_test_std=sc.transform(y_test)


# In[ ]:


lm.score(x_test_std,y_test_std)


# In[ ]:


# the score is imporved by 12 percentage
# so next step try some other methods, like nerual network
from sklearn.neural_network import MLPRegressor
from sklearn.cross_validation import column_or_1d
x_test_std.shape


# In[ ]:


mlp=MLPRegressor(hidden_layer_sizes=(30,17),activation='relu',alpha=0.01)


# In[ ]:


mlp.fit(x_train_std,column_or_1d(y_train_std))


# In[ ]:


prediction=mlp.predict(x_test_std)


# In[ ]:


mlp.score(x_test_std,y_test_std)


# In[ ]:




