#!/usr/bin/env python
# coding: utf-8

# # 住宅販売価格を予測する
# 早川　裕樹

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# In[ ]:


df_data = pd.read_csv("../input/kc_house_data.csv")
#df_data["price"] = df_data["price"] / 10**6 #単位を100万ドルにしておく


# In[ ]:


# 1.目的変数と説明変数の関係を確認するためのグラフを作成する。
y_var = ["price"]
X_var = ["sqft_living","sqft_living15"]
df = df_data[y_var+ X_var]
display(df.head())

pd.plotting.scatter_matrix(df,alpha=0.3, s=10, figsize=(5,5))
plt.show()


# 不適切な項目除去
# （lat,longはｴﾘｱごとに区切ることで使えそう）

# In[ ]:


df_data_tmp1 = df_data.drop("id",axis=1).drop("date",axis=1).drop("lat",axis=1).drop("long",axis=1).drop("zipcode",axis=1)


# 標準化

# In[ ]:


scaler = StandardScaler()
std = pd.DataFrame(scaler.fit_transform(df_data_tmp1), columns=df_data_tmp1.columns)
std_tmp = pd.concat([df_data_tmp1["price"], std.drop(["price"], axis=1)], axis=1)


# 偏相関係数確認

# In[ ]:


df = std_tmp.drop(["price"], axis=1)
for cname in df.columns:  
    y=df[cname]
    X=df.drop(cname, axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    rsquared = regr.score(X,y)
    print((cname,":" ,1/(1-np.power(rsquared,2))))


# infとなっている項目をsqft_living以外除外

# In[ ]:


std_tmp1 = std_tmp.drop(["sqft_above", "sqft_basement"], axis=1)


# In[ ]:


import statsmodels.api as sm

count = 1
for i in range(1): # 重いので1に変更（実際には7で実施）
    combi = itertools.combinations(std_tmp1.drop("price",axis=1).columns, i+1 ,) #組み合わせを求める
    for v in combi:
        y = std_tmp1["price"]
        X = sm.add_constant(std_tmp1[list(v)])
        model = sm.OLS(y, X).fit()
        if count == 1:
            min_aic = model.aic
            min_var = list(v)
        if min_aic > model.aic:
            min_aic = model.aic
            min_var = list(v)
        count += 1
        # print("AIC:",round(model.aic), "変数:",list(v))
print("====minimam AIC====")
print((min_var,min_aic))


# In[ ]:


# AICがもっとも低かったパラメータのグラフ描画
df_data_aic = std_tmp1[["price", 'bedrooms', 'bathrooms', 'sqft_living', 'waterfront', 'view', 'grade', 'yr_built']]
pd.plotting.scatter_matrix(df_data_aic,alpha=0.3, s=10, figsize=(10,10))
plt.show()


# 交差検証

# In[ ]:


y = std_tmp1["price"].values
X = std_tmp1[['bedrooms', 'bathrooms', 'sqft_living', 'waterfront', 'view', 'grade', 'yr_built']].values

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

# 評価指標の算出
y_test = df_result["y_test"]
y_pred = df_result["y_pred"]
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(("MSE=%s"%round(mse,3) ))
print(("RMSE=%s"%round(np.sqrt(mse), 3) ))
print(("MAE=%s"%round(mae,3) ))


# In[ ]:





# In[ ]:




