#!/usr/bin/env python
# coding: utf-8

# # 住宅販売価格を予測する
# House Sales in King County, USA   
# iiyama3

# ---
# ### 目次
# 1. 　関数   
# 2. 　データの読み込み   
# 3. 　データの前処理   
# 4. 　**Lasso Regression（回帰分析）**    
# 5. 　**RandomForestRegressor（ランダムフォレスト） **
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
            print((cname,X.columns[(regr.coef_> 0.5) | (regr.coef_ < -0.5)]))
        


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
        print(('k:',k,'MAE:',met,'AIC:',aic,X.columns[k]))
        


# # 2. データの読み込み

# In[2]:


# モジュールの読み込み
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_rows = 10 # 常に10行だけ表示


# In[4]:


# データの読み込み
#df000 = pd.read_csv("kc_house_data.csv") 
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


# # 4. Lasso Regression（回帰分析）
# 必要そうな特徴量だけを自動で取捨選択してくれる

# In[18]:


# データをリセット
df800 = df700
X = df800.drop(['price'],axis=1)
y = df800['price']


# In[19]:


# グリッドサーチ
from sklearn.model_selection import train_test_split                # データ分割用
from sklearn.model_selection import KFold                           # 交差検証用
from sklearn.model_selection import GridSearchCV # グリッドサーチ
from sklearn.metrics import confusion_matrix

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# 交差検証用に分割
kf = KFold(n_splits=5, random_state=1234, shuffle=True)
df_result = pd.DataFrame()
model001 = []


from sklearn.linear_model import Lasso # Lasso回帰用
# 優れたハイパーパラメータを見つけたいモデル
model001 = Lasso() 

# 試行するハイパーパラメータ
parms1 = [
    {"alpha":np.logspace(-3,1,100)},
]

grid_search = GridSearchCV(model001,            # モデルを渡す
                           param_grid = parms1, # 試行してほしいパラメータを渡す
                           cv=10,               # 汎化性能を調べる
                          )
grid_search.fit(X,y) # グリッドサーチにハイパーパラメータを探す

print((grid_search.best_score_))  # 最も良かったスコア
print((grid_search.best_params_))  # 上記を記録したパラメータの組み合わせ
print((grid_search.best_estimator_.get_params()))


# In[20]:


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

    regr = Lasso(alpha=10.0, max_iter=1000, copy_X=True) #  Lassoを適用（ハイパーパラメーターにをセット）
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
    print(("**** Training set score( {} ):  MSE={:.3f}  RMSE={:.3f}  MAE={:.3f}  Score={:.3f} ****".format(i,round(mse,3),round(np.sqrt(mse), 3),round(mae,3),regr.score(X_train, y_train))))


# In[21]:


#--------------------------------------------
# 交差検証：テスト実施
#--------------------------------------------
z = 1 # 訓練で一番良かったものをセット
y_pred = models[z].predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(("**** Test     set score( {} ):  MSE={:.3f}  RMSE={:.3f}  MAE={:.3f}  Score={:.3f} ****".format(z,round(mse,3),round(np.sqrt(mse), 3),round(mae,3),regr.score(X_test, y_test))))
print(("**** Number of features used: {} ****".format(np.sum(regr.coef_ != 0))))


# # 5. RandomForest（ランダムフォレスト）
# 
# パラメーター指定なし

# In[22]:


# データをリセット
df800 = df700
X = df800.drop(['price'],axis=1)
y = df800['price']
base_model = []


# In[23]:


from sklearn.ensemble import RandomForestRegressor # RandomForestライブラリ
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
base_model = []

for i,(train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    X_train_train, X_train_val = X_train[train_index], X_train[val_index]
    y_train_train, y_train_val = y_train[train_index], y_train[val_index]

    regr = RandomForestRegressor() 
    regr.fit(X_train_train, y_train_train)
    base_model.append(regr)
    y_pred = regr.predict(X_train_val)
    df999 = pd.DataFrame({"y_val": y_train_val, "y_pred": y_pred})
    df_result = pd.concat([df_result, df999], axis=0)
    
# validation dataによる評価指標の算出
    y_val = df_result["y_val"]
    y_pred = df_result["y_pred"]
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    print(("**** Training set score( {} ):  MSE={:.3f}  RMSE={:.3f}  MAE={:.3f}  Score={:.3f} ****".format(i,round(mse,3),round(np.sqrt(mse), 3),round(mae,3),regr.score(X_train, y_train))))


# In[25]:


#--------------------------------------------
# 交差検証：テスト実施
#--------------------------------------------
z = 0 # 訓練で一番良かったものをセット
y_pred = base_model[z].predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(("**** Test     set score( {} ):  MSE={:.3f}  RMSE={:.3f}  MAE={:.3f}  Score={:.3f} ****".format(z,round(mse,3),round(np.sqrt(mse), 3),round(mae,3),regr.score(X_test, y_test))))
print('Parameters currently in use:')

from pprint import pprint
pprint(regr.get_params())


# 過学習気味？なのでチューニングしてみる

# # 5. RandomForest（ランダムフォレスト）
# 
# * n_estimators =フォレスト内の樹木の数
# * max_features =ノードの分割に考慮されるフィーチャの最大数
# * max_depth =各決定木のレベルの最大数
# * min_samples_split =ノードが分割される前にノードに配置されたデータポイントの最小数
# * min_samples_leaf =リーフノードで許容されるデータポイントの最小数
# * bootstrap =データポイントをサンプリングする方法（置換の有無にかかわらず）

# In[26]:


# データをリセット
df800 = df700
X = df800.drop(['price'],axis=1)
y = df800['price']
base_model = []


# In[35]:


# グリッドサーチ
from sklearn.model_selection import train_test_split                # データ分割用
from sklearn.model_selection import KFold                           # 交差検証用
from sklearn.model_selection import GridSearchCV # グリッドサーチ
from sklearn.metrics import mean_squared_error, mean_absolute_error #MAE,MAE用
from sklearn.metrics import confusion_matrix


# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# 交差検証用に分割
kf = KFold(n_splits=5, random_state=1234, shuffle=True)
df_result = pd.DataFrame()

# チューニングしたいモデル
from sklearn.ensemble import RandomForestRegressor # RandomForestライブラリ
base_model = RandomForestRegressor() 

# 試行するハイパーパラメータ
random_grid = {
    'n_estimators':[10, 100, 200, 400],
    'max_depth':[1, 9, 15],
    'min_samples_leaf':[3, 5, 9],
    'min_samples_split':[3, 5, 9],
    'bootstrap':[True, False],
    'n_jobs': [-1],
}
#pprint(random_grid)
print("-- GridSearch --")

# -------- パラメーターチューニング
grid_search = GridSearchCV(base_model,            # モデルを渡す
                           random_grid, # 試行してほしいパラメータを渡す
                           cv=3               # 汎化性能を調べる
                          )
grid_search.fit(X,y) # グリッドサーチにハイパーパラメータを探す

print((grid_search.best_score_))  # 最も良かったスコア
print((grid_search.best_params_))  # 上記を記録したパラメータの組み合わせ
pprint(grid_search.best_estimator_.get_params())


# In[49]:


# 交差検証
from sklearn.ensemble import RandomForestRegressor # RandomForestライブラリ
from sklearn.metrics import mean_squared_error, mean_absolute_error #MAE,MAE用
from sklearn.model_selection import KFold                           # 交差検証用
from sklearn.model_selection import train_test_split                # データ分割用

# 学習データとテストデータに分割
X_train,X_test,y_train,y_test = train_test_split(np.array(X),np.array(y),test_size=0.2,random_state=42)
kf = KFold(n_splits=5, random_state=1234, shuffle=True)
df_result = pd.DataFrame()
base_model = []

for i,(train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    X_train_train, X_train_val = X_train[train_index], X_train[val_index]
    y_train_train, y_train_val = y_train[train_index], y_train[val_index]
    
    regr = RandomForestRegressor(
        bootstrap = True,
        criterion = 'mse',
        max_depth = 7,          # １５ -> 
        max_features = 'auto',
        max_leaf_nodes = None,
        min_impurity_decrease = 0.0,
        min_impurity_split = None,
        min_samples_leaf = 3,
        min_samples_split = 5,
        min_weight_fraction_leaf = 0.0,
        n_estimators = 400,
        n_jobs = -1,
        oob_score = False,
        random_state = None,
        verbose = 0,
        warm_start = False,
    ) 
    regr.fit(X_train_train, y_train_train)
    base_model.append(regr)
    y_pred = regr.predict(X_train_val)
    df999 = pd.DataFrame({"y_val": y_train_val, "y_pred": y_pred})
    df_result = pd.concat([df_result, df999], axis=0)
# validation dataによる評価指標の算出
    y_val = df_result["y_val"]
    y_pred = df_result["y_pred"]
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    print(("**** Training set score( {} ):  MSE={:.3f}  RMSE={:.3f}  MAE={:.3f}  Score={:.3f} ****".format(i,round(mse,3),round(np.sqrt(mse), 3),round(mae,3),regr.score(X_train, y_train))))


# In[50]:


#--------------------------------------------
# 交差検証：テスト実施
#--------------------------------------------
z = 0 # 訓練で一番良かったものをセット
y_pred = base_model[z].predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(("**** Test     set score( {} ):  MSE={:.3f}  RMSE={:.3f}  MAE={:.3f}  Score={:.3f} ****".format(z,round(mse,3),round(np.sqrt(mse), 3),round(mae,3),regr.score(X_test, y_test))))
print('Parameters currently in use:')

from pprint import pprint
pprint(regr.get_params())


# RandomForestは高いスコアを出せるが、過学習にならないためのチューニングが必要なのかなぁ

# In[ ]:




