#!/usr/bin/env python
# coding: utf-8

# # House Sales in King County, USA
# # Investigation of relationship of housing price
# Author：Kei Osanai

# ## 1. Data confirmation

# In[5]:


# class import
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import linear_model 


# In[6]:


df_data = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
df_data["price"] = df_data["price"] / 10**6 # The unit is $ 1 million
df_data.head()


# In[7]:


y_var = "price"
X_var = ["bedrooms","bathrooms","sqft_living","sqft_lot"]
df = df_data[[y_var]+X_var]
pd.plotting.scatter_matrix(df,alpha=0.3,s=10, figsize=(10,10))#散布図の作成
plt.show()#グラフをここで描画させるための行


# In[8]:


y_var = "price"
X_var = ["floors","waterfront","view","condition"]
df = df_data[[y_var]+X_var]
pd.plotting.scatter_matrix(df,alpha=0.3,s=10, figsize=(10,10))
plt.show()


# In[9]:


y_var = "price"
X_var = ["grade","sqft_above","sqft_basement","yr_built"]
df = df_data[[y_var]+X_var]
pd.plotting.scatter_matrix(df,alpha=0.3,s=10, figsize=(10,10))
plt.show()


# In[10]:


y_var = "price"
X_var = ["yr_renovated","zipcode","lat","long","sqft_living15","sqft_lot15"]
df = df_data[[y_var]+X_var]
pd.plotting.scatter_matrix(df,alpha=0.3,s=10, figsize=(10,10))
plt.show()


# # 2. Data Analytics

# In[11]:


df_data.describe ()


# # ヒートマップを表示（緯度、経度）

# In[12]:


import seaborn as sns

df2 = pd.DataFrame({ 'price' : df_data["price"],
                     'lat' : round(df_data["lat"],1),
                     'long' : round(df_data["long"],1)})
df_pivot = pd.pivot_table(data=df2, values='price', 
                                  columns='long', index='lat', aggfunc=np.mean)
df_pivot.head()
ax = sns.heatmap(data=df_pivot, cmap="Reds")
ax.invert_yaxis()


# # 値段と敷地の割合で算出する。

# In[13]:


import seaborn as sns

df2 = pd.DataFrame({ 'price' : df_data["price"] / df_data["sqft_lot"],
                     'lat' : round(df_data["lat"],1),
                     'long' : round(df_data["long"],1)})
df_pivot = pd.pivot_table(data=df2, values='price', 
                                  columns='long', index='lat', aggfunc=np.mean)
df_pivot.head()
ax = sns.heatmap(data=df_pivot, cmap="Reds")
ax.invert_yaxis()


# # 水辺の家と緯度と経度

# In[14]:


import seaborn as sns

df2 = pd.DataFrame({ 'waterfront' : df_data["waterfront"],
                     'lat' : round(df_data["lat"],1),
                     'long' : round(df_data["long"],1)})
df_pivot = pd.pivot_table(data=df2, values='waterfront', 
                                  columns='long', index='lat', aggfunc=np.mean)
df_pivot.head()
ax = sns.heatmap(data=df_pivot, cmap="Reds")
ax.invert_yaxis()


# # 景色と緯度と経度

# In[15]:


import seaborn as sns

df2 = pd.DataFrame({ 'view' : df_data["view"],
                     'lat' : round(df_data["lat"],1),
                     'long' : round(df_data["long"],1)})
df_pivot = pd.pivot_table(data=df2, values='view', 
                                  columns='long', index='lat', aggfunc=np.mean)
df_pivot.head()
ax = sns.heatmap(data=df_pivot, cmap="Reds")
ax.invert_yaxis()


# # gradeと緯度と経度

# In[16]:


import seaborn as sns

df2 = pd.DataFrame({ 'grade' : df_data["grade"],
                     'lat' : round(df_data["lat"],1),
                     'long' : round(df_data["long"],1)})
df_pivot = pd.pivot_table(data=df2, values='grade', 
                                  columns='long', index='lat', aggfunc=np.mean)
df_pivot.head()
ax = sns.heatmap(data=df_pivot, cmap="Reds")
ax.invert_yaxis()


# # yr_builtと緯度と経度

# In[17]:


import seaborn as sns

df2 = pd.DataFrame({ 'yr_built' : df_data["yr_built"],
                     'lat' : round(df_data["lat"],1),
                     'long' : round(df_data["long"],1)})
df_pivot = pd.pivot_table(data=df2, values='yr_built', 
                                  columns='long', index='lat', aggfunc=np.mean)
df_pivot.head()
ax = sns.heatmap(data=df_pivot, cmap="Reds")
ax.invert_yaxis()


# # 最も地価の高い場所からの距離の差異を求める（ユークリッド距離）

# In[18]:


import math

#SQRT(POWER((緯度1 - 緯度2) / 0.0111, 2) + POWER((経度1 - 経度2) / 0.0091, 2)) 
df2 = pd.DataFrame({ 'price' : df_data["price"],
                    'sqft_living' : df_data["sqft_living"],
                    'grade' : df_data["grade"],
                    'lat' : df_data["lat"],
                    'long' : df_data["long"],
                    'distance' : np.sqrt((((47.6-df_data["lat"]).pow(2) / 0.0111) + ((-122.2-df_data["long"]).pow(2) / 0.0091)))
                   })

df2.head()
y_var = "price"
X_var = ["price","sqft_living","distance","distance"]



# # 相関係数を求める

# In[19]:


df2.corr()


# # 異常値を削除する。高級住宅は除外する。

# In[20]:


df2.boxplot(column='price')
plt.ylabel("num")
plt.show()


# # 100万ドル以上は除外

# In[21]:


import math

#SQRT(POWER((緯度1 - 緯度2) / 0.0111, 2) + POWER((経度1 - 経度2) / 0.0091, 2)) 
df2 = pd.DataFrame({ 'price' : df_data["price"],
                    'sqft_living' : df_data["sqft_living"],
                    'grade' : df_data["grade"],
                    'lat' : df_data["lat"],
                    'long' : df_data["long"],
                    'distance' : np.sqrt((((47.6-df_data["lat"]).pow(2) / 0.0111) + ((-122.2-df_data["long"]).pow(2) / 0.0091)))
                   })
df2 = df_data[df_data["price"] < 1.0]
df2.boxplot(column='price')
plt.ylabel("num")
plt.show()


# In[22]:


df2 = pd.DataFrame({ 'price' : df2["price"],
                     'lat' : round(df2["lat"],1),
                     'long' : round(df2["long"],1)})
df_pivot = pd.pivot_table(data=df2, values='price', 
                                  columns='long', index='lat', aggfunc=np.mean)
df_pivot.head()
ax = sns.heatmap(data=df_pivot, cmap="Reds")
ax.invert_yaxis()


# # 重回帰分析

# In[23]:


from sklearn.model_selection import train_test_split

df2 = pd.DataFrame({ 'price' : df_data["price"],
                    'sqft_living' : df_data["sqft_living"],
                    'grade' : df_data["grade"],
                    'lat' : df_data["lat"],
                    'long' : df_data["long"],
                    'distance' : np.sqrt((((47.6-df_data["lat"]).pow(2) / 0.0111) + ((-122.2-df_data["long"]).pow(2) / 0.0091)))
                   })
df2 = df2[df2["price"] < 1.0]
#df2 = df2[df2["long"] < -121.6]
#df2 = df2[df2["lat"] > 47.25]


# In[24]:


df2.corr()


# In[25]:


y_var = "price"
X_var = ["distance","lat","grade","sqft_living"]
X = df2[X_var].as_matrix()
y = df2[y_var].values

# 利用できるデータのうち、学習用を8割、テスト用を2割にする
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# 学習
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(("決定係数=%s"%regr.score(X,y)))
print(("MSE=%s"%round(mse,3) ))
print(("RMSE=%s"%round(np.sqrt(mse), 3) ))
print(("MAE=%s"%round(mae,3) ))


# # サポートベクターマシン

# In[26]:


y = df2["price"].values
X = df2[['distance', 'lat', 'grade', 'sqft_living']].values

#MSE=0.014
#RMSE=0.12
#MAE=0.093


# In[ ]:





# In[ ]:




