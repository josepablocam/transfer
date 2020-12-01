#!/usr/bin/env python
# coding: utf-8

# # 最終課題（House Sales in King County, USA）
# - 【URL】
#  https://www.kaggle.com/harlfoxem/housesalesprediction/data
# 

# ### Name ：　富永（Tominaga）

# ## 今回の分析は、以下のような流れで行った。
#  1. データの確認（記述統計量、欠損値の確認）
#  1. 目的変数（price）と関係がありそうな説明変数の抽出
#  1. 異常値の確認
#  1. 異常値の削除・補完
#  1. 重回帰分析・サポートベクタマシンによるモデルの作成

# ## データの確認
# - はじめに、どのようなデータがが含まれているのか、確認を行う。

# |カラム|説明|型|
# |:--:|:--:|:--:|
# |id|a notation for a house|Numeric|
# |date|Date house was sold|String|
# |price|Price is prediction target|Numeric|
# |bedrooms|Number of Bedrooms/House|Numeric|
# |bathrooms|Number of bathrooms/bedrooms|Numeric|
# |sqft_living|square footage of the home|Numeric|
# |sqft_lot|square footage of the lot|Numeric|
# |floors|Total floors (levels) in house|Numeric|
# |waterfront|House which has a view to a waterfront|Numeric
# |view|Has been viewed|Numeric|
# |condition|How good the condition is ( Overall )|Numeric|
# |grade|overall grade given to the housing unit, based on King County grading system|Numeric|
# |sqft_above|square footage of house apart from basement|Numeric|
# |sqft_basement|square footage of the basement|Numeric|
# |yr_built|Built Year|Numeric|
# |yr_renovated|Year when house was renovated|Numeric|
# |zipcode|zip|Numeric|
# |lat|Latitude coordinate|Numeric|
# |long|Longitude coordinate|Numeric|
# |sqft_living15|Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area|Numeric|
# |sqft_lot15|lotSize area in 2015(implies-- some renovations)|Numeric|

# - 英語での説明がわからなかったので、日本語にしてみました。
# 
# |カラム|説明|型|
# |:--:|:--:|:--:|
# |id|識別子|Numeric|
# |date|売却された日|String|
# |price|価格（目的変数）|Numeric|
# |bedrooms|ベッドルームの数/家|Numeric|
# |bathrooms|お風呂の数/家|Numeric|
# |sqft_living|家の平方フィート（広さ）|Numeric|
# |sqft_lot|区画(lot)の平方フィート|Numeric|
# |floors|家の中の全フロア（レベル）|Numeric|
# |waterfront|海辺（waterfront）を望む家|Numeric
# |view|内見された回数？|Numeric|
# |condition|どのくらいの状態が良いか（全体的）|Numeric|
# |gradeloveral | 「King County grading system」に基づいて、住宅部門に与えられた格付け|Numeric|
# |sqft_above|地下室（basement）を含まない家の平方フィート|Numeric|
# |sqft_basement|地下室（basement）の平方フィート|Numeric|
# |yr_built|家が建った年|Numeric|
# |yr_renovated|家が改築された年|Numeric|
# |zipcode|郵便番号|Numeric|
# |lat|緯度座標（Latitude coordinate）|Numeric|
# |long|経度座標（Longitude coordinate）|Numeric|
# |sqft_living15|2015年のリビングルーム面積（いくつかの改装を含む）これはロットサイズの面積に影響を与えているかもしれない|Numeric|
# |sqft_lot15|2015年のロットサイズ面積（いくつかの改装を含む）|Numeric|

# In[ ]:


#ライブラリの読み込み
import pandas as pd
from IPython.display import display
from dateutil.parser import parse
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV


# In[ ]:


#データの読み込み
df_data = pd.read_csv("../input/kc_house_data.csv")

print("")
print("データセットの頭出し")
display(df_data.head())


# In[ ]:


# date列の変換（日付の形に変更） (説明変数として使わないため、実行しない。)
#df_data["date"] = [ parse(i[:-7]).date() for i in df_data["date"]]
#display(df_data.head())


# ## 欠損値の確認

# In[ ]:


# 欠損値のデータが含まれているかどうか確認する
pd.DataFrame(df_data.isnull().sum(), columns=["num of missing"])


# - 上記の表より、欠損値はなし。

# In[ ]:


#不要な列の削除
df_data_main = df_data.drop(["id","date","zipcode"], axis=1)
df1 = df_data_main.iloc[:,:9]
display(df1.head())
df2 = df_data_main.iloc[:,[0]+list(range(9,18))]
display(df2.head())


# In[ ]:


# describe（記述統計量の算出）
df_data.describe()


# In[ ]:


# 散布図行列
pd.plotting.scatter_matrix(df1,figsize=(10,10))
plt.show()
pd.plotting.scatter_matrix(df2,figsize=(10,10))
plt.show()


#  - 上記のグラフが小さくて見えにくかったため、１つずつグラフを描画。

# In[ ]:


li_combi = list(itertools.combinations(df_data_main.columns[0:], 2))
for X,Y in li_combi:
    if X=='price':
        print("X=%s"%X,"Y=%s"%Y)
        df_data_main.plot(kind="scatter",x=X,y=Y,alpha=0.7,s=10)#散布図の作成
        plt.xlabel(X)
        plt.ylabel(Y)
        plt.tight_layout()
        plt.show()#グラフをここで描画させるための行


# ### 相関関係の確認

# In[ ]:


df_data_main.corr()


# 上記の分散図と、相関関係の表より、次の変数を用いることとする。
# 
# 1. sqft_living
# 1. grade
# 1. sqft_above
# 1. bathrooms
# 
# 順番は、priceと各説明変数の相関係数が大きい順となっている。
# なお、「sqft_living15」も、比較的高い相関関係を示しているが、「sqft_living」との関係があるため、除外。

# ## 異常値の検討

# ### ヒストグラムによる確認

# In[ ]:


for col in df_data_main.columns:
    print(col)
    df_data_main[col].hist()
    plt.xlabel(col)
    plt.ylabel("num")
    plt.show()


# - データ数が多く、ヒストグラムでは、異常値が判定できなかったため、次に箱ひげ図による確認を試みる。

# ### 箱ひげ図による確認

# In[ ]:


for col in df_data_main.columns:
    print(col)
    df_data_main.boxplot(column=col)
    plt.xlabel(col)
    plt.ylabel("num")
    plt.show()


# ## 異常値の除外

# In[ ]:


#異常値を除外
def drop_outlier(df):
  for i, col in df.iteritems():
    #四分位数
    q1 = col.describe()['25%']
    q3 = col.describe()['75%']
    iqr = q3 - q1 #四分位範囲

    #外れ値の基準点
    outlier_min = q1 - (iqr) * 1.5
    outlier_max = q3 + (iqr) * 1.5

    #範囲から外れている値を除く
    col[col < outlier_min] = None
    col[col > outlier_max] = None

if __name__ == '__main__':
    drop_outlier(df_data_main)
    
    for col in df_data_main.columns:
        print(col)
        df_data_main.boxplot(column=col)
        plt.xlabel(col)
        plt.ylabel("num")
        plt.show()


# 除外した異常値を補うための処理。

# In[ ]:


#どの程度、異常値を除外したのか、確認を行う。
df_data_main.isnull().any(axis=0)


# In[ ]:


#全体の平均で、欠損値を埋める。
for col in df_data_main.columns:
    mean_all = df_data_main[col].mean()
    df_data_main[col] = df_data_main[col].fillna(mean_all)
    #df_data_main.loc[parse(col):parse(col + 'after')]
df_data_main


# ### １.重回帰分析

# In[ ]:


#分析に用いるデータのみを取得。
X_var = ["sqft_living","grade","sqft_above","bathrooms"]
y_var = ["price"]
df = df_data[y_var+ X_var]

# scikit learnの入力形式に変換する
X = df[X_var].as_matrix()
y = df[y_var].values

# 学習
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X, y)

print("決定係数=",regr.score(X,y))


# In[ ]:


from sklearn.model_selection import train_test_split
# 利用できるデータのうち、学習用を8割、テスト用を2割にする
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE=%s"%round(mse,3) )
print("RMSE=%s"%round(np.sqrt(mse), 3) )
print("MAE=%s"%round(mae,3) )


# ### 2.サポートベクターマシン

# In[ ]:


# 標準化
stdsc = StandardScaler()
X_train_transform = stdsc.fit_transform(X_train)
X_test_transform = stdsc.transform(X_test)

print(X_train_transform)
print(X_test_transform)

# SVMの実行
clf = SVR(C=5, kernel="linear")
clf.fit(X_train_transform, y_train)

# 未知のデータを識別する
clf.predict(X_test_transform)


# In[ ]:




