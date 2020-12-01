#!/usr/bin/env python
# coding: utf-8

# # 最終課題
# ### 氏名：とみなが
# ### 選んだ課題【House Sales in King County, USA】住宅販売価格を予測する
# - 【URL】
#  https://www.kaggle.com/harlfoxem/housesalesprediction/data
# 

# ## 中間発表（Day5までの宿題）
# -  タイトル
# -  氏名もしくはニックネーム
# -  選んだ課題
# -  目的変数と説明変数の関係を確認するためのグラフ。また、そのグラフからわかるこ とを文章で。
# -  目的変数を説明するのに有効そうな説明変数。また、それらが有効だと考えた理由を 文章で。
# -  欠測値と異常値を確認した結果。また、欠測値や異常値をが存在する場合は、その処 理方法。
# -  使えそうなアルゴリズムの候補。

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


# -*- coding: utf-8 -*-
#ライブラリの読み込み
import pandas as pd
from IPython.display import display
from dateutil.parser import parse
import matplotlib.pyplot as plt


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


# In[ ]:


import itertools
li_combi = list(itertools.combinations(df_data_main.columns[0:], 2))
for X,Y in li_combi:
    if X=='price':
        print(("X=%s"%X,"Y=%s"%Y))
        df_data_main.plot(kind="scatter",x=X,y=Y,alpha=0.7,s=10,c="price",colormap="winter")#散布図の作成
        plt.xlabel(X)
        plt.ylabel(Y)
        plt.tight_layout()
        plt.show()#グラフをここで描画させるための行


# ### 相関関係の確認

# In[ ]:


df_data_main.corr()


# 上記の分散図と、相関関係の表より、次の変数を用いることとする(順番あり)。
# 1. sqft_living
# 1. grade
# 1. sqft_above
# 1. bathrooms
# 
# ※「sqft_living15」も、比較的高い相関関係を示しているが、「sqft_living」との関係があるため除外。

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


# In[ ]:


#異常値を除外したグラフを描画する。
for col in df_data_main.columns:
    
    Q1 = df_data_main[col].quantile(.25)
    Q3 = df_data_main[col].quantile(.75)

    #print(Q1)
    #print(Q3)
    
    IQR = Q3 - Q1
    threshold = Q3 + 1.5*IQR

    df_outlier = df_data_main[(df_data_main[col] < threshold)]

    print(col)
    df_outlier.boxplot(column=col)
    plt.xlabel(col)
    plt.ylabel("num")
    plt.show()


# ### 今後取り組まなければいけないこと
# 1. 異常値を取り除いたデータを、データフレームワークへの格納する。
# 1. 1.で作成したデータフレームワークの、異常値への処理。
# 1. より詳細な分析（一部、まだ言葉で説明できていない部分があるため、そちらの補足を行う。）

# ### 使えそうなアルゴリズム
# - 重回帰分析
# - サポートベクタマシン
# - 決定木
# - ランダムフォレスト
