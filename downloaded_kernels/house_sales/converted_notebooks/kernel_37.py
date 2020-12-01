#!/usr/bin/env python
# coding: utf-8

# # 課題：House Sales in King County, USA
# 名前：佐々木知美

# ##### データの概要 https://www.kaggle.com/harlfoxem/housesalesprediction/data
# 目的：住宅価格の予測 / 目的変数：price
# 
# |項目|データ詳細|
# |-|-|
# |期間|2014年5月～2015年5月（1年間）|
# |種別|購入住宅データ|
# |地域|アメリカ：ワシントン・シアトルエリア|
# |サンプル数|21,613 s|

# # ◆実施したこと

# ### **【１】データの前処理・アナライズ**
# 
# #### 　　　データ：priceの対数化
# 
# #### 　　　新変数の作成：northエリアの判別
# 
# ### **【２】変数の選択**
# 
# #### 　　　方法：AIC値による選択
# 
# ### **【３】勾配降下法**
# 

# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

# # 【１】データの前処理・アナライズ

# #### ◆データの確認

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from IPython.display import display
from dateutil.parser import parse
import matplotlib.pyplot as plt
from IPython.core.display import display
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA #主成分分析用ライブラリ
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D #3D散布図の描画
import itertools #組み合わせを求めるときに使う
from sklearn.linear_model import LinearRegression
import seaborn as sns


# In[ ]:


# データの読み込み
df_data = pd.read_csv("../input/kc_house_data.csv")
print(df_data.columns)
display(df_data.head())
display(df_data.tail())


# In[ ]:


# coutn missing
pd.DataFrame(df_data.isnull().sum(), columns=["num of missing"])
# 欠損データなし


# In[ ]:


# データの型確認
df_data.info()


# In[ ]:


# データの数値の個数
print(df_data.shape)
print(df_data.nunique())


# In[ ]:


# date列の変換
df_data["date"] = [ parse(i[:-7]).date() for i in df_data["date"]]
display(df_data.head())


# ________
# #### **◆基礎統計**

# In[ ]:


# 基礎統計
df_data.describe().round(1)


# In[ ]:


# priceのhistogram
ax=df_data['price'].hist(rwidth=100000,bins=20)
ax.set_title('price')
plt.show()


# In[ ]:


df_en=df_data.drop(['id','date'],axis=1)
df_en1=df_data.drop(['id','date'],axis=1)


# In[ ]:


# priceを対数で確認
s_price_log = np.log(df_en1['price'])
s_price_log.plot.hist(x='price')


# ##### **＝＝＝【Point】ばらけたため、データは対数を採用**

# In[ ]:


# priceの対数化
df_log= df_en1
df_log["price"] = df_en1["price"].apply( lambda x: np.log(x) )


# In[ ]:


# 基礎統計（price対数データ）
df_en1.describe().round(1)


# ________
# #### **◆目的変数との関係性確認**

# In[ ]:


# price（対数データ）と全変数の掛け合わせ
cols = [x for x in df_en1.columns if x not in ('id', 'price', 'date')]
fig, axes = plt.subplots(len(cols), 2, figsize=(10,100))
for i, col in enumerate(cols):
    df_en1[col].plot.hist(ax=axes[i, 0])
    df_en1.plot.scatter(x=col, y = 'price', ax=axes[i, 1])


# In[ ]:


# 全変数同士の相関の確認
cor = df_en1.corr().style.background_gradient().format("{:.2f}")
cor 


# ________
# ###### 【Point】
# ######   ＝＝＝＝①目的変数priceとの相関が高い項目
# 
# |順位|項目|相関係数|
# |-|-|-|-|
# |1位|grade|0.70|
# |1位|sqft_living|0.70|
# |3位|sqft_living15|0.62|
# |4位|sqft_above|0.60|
# |5位|bathrooms|0.55|
# 
# ######   ＝＝＝＝②lat,long,zipcode・・・latのみ相関がやや高い。エリアに関するデータであるため、関係性を深堀り。
# _______

# #####   ◆　**lat,long,zipcodeのエリアデータの深堀り**

# In[ ]:


#lat,long,priceの関係性可視化

X = df_en1["lat"]
Y = df_en1["long"]
Z = df_en1["zipcode"]

fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel("lat")
ax.set_ylabel("long")
ax.set_zlabel("zipcode")

ax.scatter3D(X,Y,Z)
plt.show()


# ###### **＝＝＝【Point】ある一定エリアの土地がzipcode5桁のみでは判明できない　⇒　lat,longのみで利用**

# # ★★★Attention point

# In[ ]:


# lat、long確認（priceをカラーリング）
plt.figure(figsize = (15,10))
g = sns.FacetGrid(data=df_data, hue='price',size= 5, aspect=2)
g.map(plt.scatter, "long", "lat")
plt.show()


# ###### **＝＝＝【Point】大きく見ると、northエリアのほうが価格が高い　⇒　latを47.5地点から、南北に分ける**

# In[ ]:


# northエリア判別の新変数作成
north_array = np.zeros((df_en.shape[0],1),float)

for i in range(df_en.shape[0]):
    if df_en.iat[i, 15] < 47.5000 and df_en.iat[i, 15] >= 47.1000:
        north_array[i, 0] = 0
    elif df_en.iat[i, 15] < 47.8000 and df_en.iat[i, 15] >= 47.5000:
        north_array[i, 0] = 1
        
north_array_df = pd.DataFrame(north_array)
north_array_df.columns = ["north"]
print(north_array_df)


# In[ ]:


# データ合体
df_en = pd.concat([df_en,north_array_df], axis=1)
df_en1 = pd.concat([df_en1,north_array_df], axis=1)
print(df_en.columns)
print(df_en1.columns)


# 
# ________
# # 【２】変数の選択

# In[ ]:


#相関確認
cor = df_en1.corr().style.background_gradient().format("{:.2f}")
cor 

# ★north（0.52）のほうが、元のlat（0.45）より説明力がUPしたので、変数として採用


# In[ ]:


#　zipcode,latおよび、多重共線性が出たsqft_above,sqft_basementを除外
df_en=df_en.drop(['sqft_above','sqft_basement','zipcode','lat'],axis=1)
df_en1=df_en1.drop(['sqft_above','sqft_basement','zipcode','lat'],axis=1)
print(df_en.columns)
print(df_en1.columns)


# In[ ]:


#多重共線性の確認
from sklearn.linear_model import LinearRegression
df_vif = df_en.drop(["price"],axis=1)
for cname in df_vif.columns:  
    y=df_vif[cname]
    X=df_vif.drop(cname, axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    rsquared = regr.score(X,y)
    #print(cname,":" ,1/(1-np.power(rsquared,2)))
    if rsquared == 1:
        print(cname,X.columns[(regr.coef_> 0.5) | (regr.coef_ < -0.5)])


# In[ ]:


# 変数の選択
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm 

count = 1
for i in range(5):
    combi = itertools.combinations(df_en1.drop(["price"],axis=1).columns, i+1) 
    for v in combi:
        y = df_en["price"]
        X = sm.add_constant(df_en[list(v)])
        model = sm.OLS(y, X).fit()
        if count == 1:
            min_aic = model.aic
            min_var = list(v)
        if min_aic > model.aic:
            min_aic = model.aic
            min_var = list(v)
        count += 1
        print("AIC:",round(model.aic), "変数:",list(v))
print("====minimam AIC====")
print(min_var,min_aic)

# ★　====minimam AIC====['sqft_living', 'waterfront', 'grade', 'yr_built', 'north'] 590053.189844908


# # ★★★Attention point

# In[ ]:


# LinerRegresshionで、選択した説明変数の決定係数および、説明変数の傾きを確認
y=df_en1["price"].values
X=df_en1[['sqft_living', 'waterfront', 'grade', 'yr_built', 'north']].values
regr = LinearRegression(fit_intercept=True)
regr.fit(X, y)
print("決定係数=%s"%regr.score(X,y))
print("傾き=%s"%regr.coef_,"切片=%s"%regr.intercept_)


# ________
# ##### **【Point】**
# #####  **＝＝＝①最も傾きが大きいのは、waterfront。次いでnorth、grade**
# ##### **＝＝＝ ②yr_builtは負の傾きになっているため、下記仮説をデータにて確認**
# 
# 　　　　仮説a：yr_builtは、新しいほどpriceが上がるのではないか？
# 
#  　 　　　　 （ただしアメリカは、日本のように新築住宅のほうが価値がある、という文化ではないと言われている※古いほうがデザインがオーソドックスでいい、など）
#  
# 
# 　　　　仮説b：yr_builtは、yr_renovatedと組み合わせた変数を作成するべきではないか？
# 
#  　  　　　　（ただしアメリカは、業者にリノベーションを頼むのではなく、個人が日常的に手入れをする文化だと言われている）

# In[ ]:


# yr_builtデータ確認
plt.figure(figsize = (15,10))
g = sns.FacetGrid(data=df_data,hue='price',size= 10, aspect=2)
g.map(plt.scatter, "yr_built", "yr_renovated")
plt.show()

# 仮説a：yr_builtが新しいほどpriceが高いことが顕著に表れていない⇒仮説棄却
# 仮説b：リノベーションをした住宅のほうが価格が高いわけではない⇒仮説棄却
# どちらの仮説も棄却したため、yr_builtはこのままでOK


# 
# ________
# # 【３】勾配降下法
# # ★★★Attention point

# In[ ]:


# 勾配降下法のcross validationによる検証
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

X_train,X_test,y_train,y_test = train_test_split(np.array(X),np.array(y),test_size=0.3,random_state=1234)

kf = KFold(n_splits=5, random_state=1234, shuffle=True)

df_result = pd.DataFrame()
models = []

for i,(train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    X_train_train, X_train_val = X_train[train_index], X_train[val_index]
    y_train_train, y_train_val = y_train[train_index], y_train[val_index]
    
    regr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1,
     max_depth=1, random_state=0, loss='ls')
    
    regr.fit(X_train_train, y_train_train)
    models.append(regr)
    y_pred = regr.predict(X_train_val)
    df = pd.DataFrame({"y_val":y_train_val, "y_pred":y_pred})
    df_result = pd.concat([df_result, df], axis=0)

# validation dataによる評価指標の算出
    y_val = df_result["y_val"]
    y_pred = df_result["y_pred"]
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred) # ここだけとりあえず見る！
    print(i)
    print("MSE=%s"%round(mse,3) )
    print("RMSE=%s"%round(np.sqrt(mse), 3) )

import numpy as np
import matplotlib.pyplot as plt
print("MAE=%s"%round(mae,3) )


# In[ ]:


#　モデルの精度評価
y_pred = models[1].predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE=%s"%round(mse,3) )
print("RMSE=%s"%round(np.sqrt(mse), 3) )
print("MAE=%s"%round(mae,3) )


# ________
# ##### 【Point】
# #####  ＝＝＝パラメーターは、estimatorsが1000～1500付近で最高値で、それ以上増やすと下がったため、1000に設定。
# #####  ＝＝＝そのほかのパラメーターはほとんど変化が見られなかった。
# ________
