#!/usr/bin/env python
# coding: utf-8

# # 住宅販売価格を予測する
# House Sales in King County, USA   
# kiiyama

# ## 前回までとの違い   
# ---
# ### 目次
# 1. 　関数   
# 2. 　データの読み込み   
# 3. 　データの前処理   
# 4. 　**GridSearchCV（ランダムフォレスト） … フリーズするため一時中断**    
# 5. 　**Lasso Regression（回帰分析）**    
# 6. 　**Ensemble regressor（決定木のバギング）**   
#    
# ### 説明（ココを追加しました）
# * **前半の 1. 2. 3.は、前回までと同じ内容ですので読み飛ばしていただいて構いません**
# * **4. のグリッドサーチ（ランダムフォレスト）は、学習時にフリーズしてしまうので、一旦コメントにしています**
# * **今回新しく 5. 6.を追加しました。アルゴリズムの適用数を絞りましたが、とりあえずの実装で理解はこれからです...**
# 

# # 1. 関数

# In[1]:


###############################
### マルチコの検出 VIFの計算
###############################
def fc_vif(dfxxx):
    from sklearn.linear_model import LinearRegression
    df_vif = dfxxx.drop(["price"],axis=1)
    for cname in df_vif.columns:
        y=df_vif[cname]
        X=df_vif.drop(cname, axis=1)
        regr = LinearRegression(fit_intercept=True)
        regr.fit(X, y)
        rsquared = regr.score(X,y)
        #print(cname,":" ,1/(1-np.power(rsquared,2)))
        if rsquared == 1:
            print(cname,X.columns[(regr.coef_> 0.5) | (regr.coef_ < -0.5)])
        


# In[2]:


###############################
### 変数の選択 MAE:AIC
###############################
def fc_var(X, y):
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.feature_selection import SelectKBest,f_regression
    
    N = len(X)
    
    for k in range(1,len(X.columns)):
        skb = SelectKBest(f_regression,k=k).fit(X,y)
        sup = skb.get_support()
        X_selected = X.transpose()[sup].transpose()
        regr = linear_model.LinearRegression()
        model = regr.fit(X_selected,y)
        met = mean_absolute_error(model.predict(X_selected),y)
        aic = N*np.log((met**2).sum()/N) + 2*k
        print('k:',k,'MAE:',met,'AIC:',aic,X.columns[k])
        


# # 2. データの読み込み

# In[3]:


# モジュールの読み込み
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_rows = 10 # 常に10行だけ表示


# In[4]:


# データの読み込み
df000 = pd.read_csv("../input/kc_house_data.csv") 
display(df000.head())


# # 3. データの前処理

# In[5]:


df600 = df000.drop(['date'],axis=1) #dataの削除
#相関係数表示
df600.corr().style.background_gradient().format("{:.2f}") # わかりやすく色付け表示


# In[6]:


# マルチコの検出 VIFの計算
rc = fc_vif(df600)


# #### sqft_basementを削除
# 理由 sqft_basement + sqft_above = sqft_living のため強相関であり、
# また、sqft_basementには"0"が含まれるため

# In[7]:


df700 = df600.drop(['sqft_basement','yr_renovated','zipcode','id'],axis=1)

for c in df700.columns: # 列の分だけ繰り返す
    if (c != "price") & (c != "date"): # ただし、price自身と日付は除く
        df000[[c,"price"]].plot(kind="scatter",x=c,y="price") # priceとの散布図


# In[8]:


# マルチコの検出 VIFの計算（再度）→　
rc = fc_vif(df700)


# マルチコは検出されなくなった

# In[9]:


df800 = df700
X = df800.drop(['price'],axis=1)
y = df800['price']

#V変数の選択
rc = fc_var(X, y)


# In[10]:


from sklearn.linear_model import LinearRegression
regr = LinearRegression(fit_intercept=True).fit(X,y)
pd.Series(regr.coef_,index=X.columns).sort_values()  .plot(kind='barh',figsize=(6,8))


# # 4. GridSearchCV　－－フリーズするため一時中断－－
# 機械学習モデルのハイパーパラメータを自動的に最適化してくれるというありがたい機能

# In[11]:


from sklearn.ensemble import RandomForestClassifier # ランダムフォレスト
from sklearn.model_selection import GridSearchCV,train_test_split # グリッドサーチ
from sklearn.metrics import confusion_matrix,classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

param_grid = [{'n_estimators':[10,20]}]
RFC = RandomForestClassifier()
cv = GridSearchCV(RFC,param_grid,verbose=0,cv=5)


# ### --- fit時点でフリーズするため、一旦、次の3行をコメントアウト ---

# In[12]:


# cv.fit(X_train,y_train) # 訓練してモデル作成


# In[13]:


# テスト
# confusion_matrix(y_test,cv.predict(X_test))


# In[14]:


# pd.Series(cv.best_estimator_.feature_importances_,index=df000.columns[24:]).sort_values().plot(kind='barh')


# # 5. Lasso Regression（回帰分析）
# 必要そうな特徴量だけを自動で取捨選択してくれる

# In[15]:


# データをリセット
df800 = df700
X = df800.drop(['price'],axis=1)
y = df800['price']


# In[16]:


from sklearn.linear_model import Lasso                       # Lasso回帰用
from sklearn.metrics import mean_squared_error, mean_absolute_error #MAE,MAE用
from sklearn.model_selection import KFold                           # 交差検証用
from sklearn.model_selection import train_test_split                # データ分割用

#--------------------------------------------
# データの整形——説明変数xの各次元を正規化
#--------------------------------------------
from sklearn import preprocessing # 正規化用
sc = preprocessing.StandardScaler()
sc.fit(X)
X = sc.transform(X)
#--------------------------------------------

# 学習データとテストデータに分割
X_train,X_test,y_train,y_test = train_test_split(np.array(X),np.array(y),test_size=0.2,random_state=42)

kf = KFold(n_splits=5, random_state=1234, shuffle=True)

df_result = pd.DataFrame()
models = []

for i,(train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    X_train_train, X_train_val = X_train[train_index], X_train[val_index]
    y_train_train, y_train_val = y_train[train_index], y_train[val_index]

    regr = Lasso(alpha=1.0) #  Lasso Regressorを適用
    regr.fit(X_train_train, y_train_train)
    models.append(regr)
    y_pred = regr.predict(X_train_val)
    df999 = pd.DataFrame({"y_val":y_train_val, "y_pred":y_pred})
    df_result = pd.concat([df_result, df999], axis=0)
    
# validation dataによる評価指標の算出
    y_val = df_result["y_val"]
    y_pred = df_result["y_pred"]
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    print("**** Training set score( {} ):  MSE={:.3f}  RMSE={:.3f}  MAE={:.3f}  Score={:.3f} ****".format(i,round(mse,3),round(np.sqrt(mse), 3),round(mae,3),regr.score(X_train, y_train)))


# In[17]:


#--------------------------------------------
# 交差検証：テスト実施
#--------------------------------------------
z = 2 # 訓練で一番良かったものをセット
y_pred = models[z].predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("**** Test     set score( {} ):  MSE={:.3f}  RMSE={:.3f}  MAE={:.3f}  Score={:.3f} ****".format(z,round(mse,3),round(np.sqrt(mse), 3),round(mae,3),regr.score(X_test, y_test)))


# # 6. Ensemble regressor（決定木のバギング）
# アンサンブル回帰によって、連続データを線形回帰分析する   
# アンサンブル回帰のBaggingRegressorを使用することで、非線形な回帰分析が可能に   

# In[18]:


# データをリセット
df800 = df700
X = df800.drop(['price'],axis=1)
y = df800['price']


# In[19]:


from sklearn.metrics import mean_squared_error, mean_absolute_error # MAE,MAE用
from sklearn.model_selection import KFold                           # 交差検証用
from sklearn.model_selection import train_test_split                # データ分割用

from sklearn.ensemble import BaggingRegressor                       # バギング 用
from sklearn.tree import DecisionTreeRegressor

# 学習データとテストデータに分割
X_train,X_test,y_train,y_test = train_test_split(np.array(X),np.array(y),test_size=0.2,random_state=42)

kf = KFold(n_splits=4, random_state=1234, shuffle=True)

df_result = pd.DataFrame()
models = []

for i,(train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    X_train_train, X_train_val = X_train[train_index], X_train[val_index]
    y_train_train, y_train_val = y_train[train_index], y_train[val_index]

    regr = BaggingRegressor(DecisionTreeRegressor(), n_estimators=100, max_samples=0.3) # バギング（決定木）
    
    regr.fit(X_train_train, y_train_train)
    models.append(regr)
    y_pred = regr.predict(X_train_val)
    df000 = pd.DataFrame({"y_val":y_train_val, "y_pred":y_pred})
    df_result = pd.concat([df_result, df000], axis=0)
    
# validation dataによる評価指標の算出
    y_val = df_result["y_val"]
    y_pred = df_result["y_pred"]
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    print("**** Training set score( {} ):  MSE={:.3f}  RMSE={:.3f}  MAE={:.3f}  Score={:.3f} ****".format(i,round(mse,3),round(np.sqrt(mse), 3),round(mae,3),regr.score(X_train, y_train)))
    


# In[20]:


#--------------------------------------------
# 交差検証：テスト実施
#--------------------------------------------
z = 3 # 訓練で一番良かったものをセット
y_pred = models[z].predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("**** Test     set score( {} ):  MSE={:.3f}  RMSE={:.3f}  MAE={:.3f}  Score={:.3f} ****".format(z,round(mse,3),round(np.sqrt(mse), 3),round(mae,3),regr.score(X_test, y_test)))


# ## コメント   
# お忙しいところ恐れ入ります。   
# とりあえずの実装で理解はこれからな状況です。。。

# In[ ]:




