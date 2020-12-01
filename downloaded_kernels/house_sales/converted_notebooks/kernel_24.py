#!/usr/bin/env python
# coding: utf-8

# ## 1. Load libraries and data: データ/ライブラリの読み込み

# In[31]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# set pandas options
pd.set_option("display.max_columns", 100)


# In[76]:


df = pd.read_csv('../input/kc_house_data.csv')

display(df.shape)
display(df.dtypes)
display(df.describe())
display(df.head(10))
display(df.tail(10))


# In[33]:


# 不要データ削除
df = df.drop("id", axis = 1)


# ## 2. Data Pre-Precessing: 前処理

# ### a. Checking for null: 欠損値の確認

# In[34]:


display(pd.DataFrame(df.isnull().sum(), columns = ["number of null"]))


# ### b. Exploratory Data Analysis: 可視化による関係把握

# 住宅販売価格を予測するため、目的変数は price として、説明変数を探す。

# In[36]:


display(df.corr().style.background_gradient().format("{:.2f}"))


# In[78]:


# price 行を抽出し、相関係数の高い順に並べ替える
display(df.corr()["price":"price"].sort_values(by="price", axis=1, ascending=False).style.format("{:.2f}"))


# price との相関係数から説明変数を検討していくと、下記の順に相関係数が高いことがわかる    
# **sqft_living > grade > sqft_above > sqft_living15 > bathrooms > view > sqft_basement > bedrooms ...**  
# 
# price との相関係数の高いデータTop5(sqft_living,grade,sqft_above,sqft_living15,bathrooms)について相関関係があるため、個別に確認する。

# In[6]:


labels = ['sqft_living','grade','sqft_above','sqft_living15','bathrooms']
y = 'price'

for i in range(len(labels)):
    plt.figure(figsize = (5,3))
    plt.scatter(df[labels[i]], df[y], s = 5)
    plt.xlabel(labels[i])
    plt.ylabel(y)


# Top5(sqft_living,grade,sqft_above,sqft_living15,bathrooms)について相関関係があるため、説明変数として採用してみる。

# ### c. Feature Engineering: 特徴量エンジニアリング

# In[7]:


# 本来実視すべきだが、ここでは一旦スキップ


# ## 3. Training Model: モデルの構築

# In[8]:


from sklearn.linear_model import LinearRegression

# 特徴量エンジニアリングなし、かつ、相関係数のTop5のみでモデルを作成
y = df["price"]
X = df[["sqft_living","grade","sqft_above","sqft_living15","bathrooms"]]

regr = LinearRegression(fit_intercept = True)
regr.fit(X, y)

print("score = %s"%regr.score(X, y))


# In[9]:


from sklearn.model_selection import train_test_split

# train_test_split関数を使用して、80%学習に、20%テストに当てる
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

regr = LinearRegression(fit_intercept=True)
regr.fit(X_train, y_train)

print("train score = %s"%regr.score(X_train, y_train))


# ## 4. Evalueation: 評価

# In[10]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_pred = regr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE = %s" % round(mse,3) )
print("RMSE = %s" % round(np.sqrt(mse), 3) )
print("MAE = %s" % round(mae,3) )


# ## 5. Discussion: 考察
# スコアがかなり低いため、説明変数の選定に問題がある模様!?  
# 特徴量エンジニアリング未実施が原因と考えられる。

# ## 6. Feature Engineering: 特徴量エンジニアリング
# ### a. Checking for multicollinearity: マルチコ検出

# In[11]:


_df = df.drop('date',axis=1)
for cname in _df.columns:  
    y = _df[cname]
    X = _df.drop(cname, axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    rsquared = regr.score(X,y)
    if rsquared == 1:
        print(cname,X.columns[(regr.coef_> 0.5) | (regr.coef_ < -0.5)])


# price との相関係数は sqft_living > sqft_above > sqft_basementの順に高いため、一番低い sqft_basementを削除してみる。

# In[12]:


_df = _df.drop('sqft_basement',axis=1)
for cname in _df.columns:  
    y = _df[cname]
    X = _df.drop(cname, axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    rsquared = regr.score(X,y)
    if rsquared == 1:
        print(cname,X.columns[(regr.coef_> 0.5) | (regr.coef_ < -0.5)])


# マルチコの発生を抑えることに成功した。  
# すでに構築済みモデルでは、sqft_basementは説明変数の対象から外していたため、マルチコは発生していないと考えられる。単に相関係数が高いからといって、説明変数に採用するだけでは精度は上がらない模様。

# ### b. Encoding Categorical Values: カテゴライズ変数のencoding

# drop した各データの相関関係を再確認する

# In[13]:


labels = ["bedrooms","sqft_lot","floors"
         ,"waterfront","view","condition","sqft_basement"
         ,"yr_built","yr_renovated","zipcode","lat"
         ,"long","sqft_lot15"]
y = 'price'

for i in range(len(labels)):
    plt.figure(figsize = (5,3))
    plt.scatter(df[labels[i]], df[y], s = 5)
    plt.xlabel(labels[i])
    plt.ylabel(y)


# dropしたデータには、カテゴライズデータが多いため、encodingを実施する。

# #### dateデータについて
# 仮説として、売れやすい月は合ってもいい（日本でも3月に引っ越しやすかったりする）　　
# 仮説として、土日の方が良く売れるだろうし、売れやすいときは少し値段を下げるかもしれないので、dowとmonthを採用する。

# In[14]:


display(df.date.head())


# In[15]:


df['date'] = pd.to_datetime(df['date'])
display(df.date.head())


# In[16]:


df['dow'] = pd.to_datetime(df.date).map(lambda x:'dow'+str(x.weekday()))
df['month'] = pd.to_datetime(df.date).map(lambda x:'month'+str(x.month))


# #### yr_renobated データについて

# In[17]:


display(df["yr_renovated"].value_counts().sort_index().head())


# 仮説として、リノベーションをしているかどうかが影響が出そう。

# In[18]:


df['yr_renovated_bin'] = np.array(df['yr_renovated'] != 0)*1
display(df['yr_renovated_bin'].value_counts().sort_index())


# #### zipcodeデータについて
# zipcodeは郵便番号、そのまま回帰アルゴリズムに入れてはまずいと考えられる。

# In[19]:


df['zipcode_str'] = df['zipcode'].astype(str).map(lambda x:'zip_'+x)


# #### カテゴライズ変数へone hot encoding

# In[20]:


df_en = pd.concat([df, pd.get_dummies(df['zipcode_str'])], axis = 1)
df_en = pd.concat([df_en, pd.get_dummies(df.dow)], axis = 1)
df_en = pd.concat([df_en, pd.get_dummies(df.month)], axis = 1)


# In[21]:


display(df_en.head())


# In[22]:


#　特徴量エンジニアリングしたのでdrop
df_en_fin = df_en.drop(['date', 'zipcode', 'yr_renovated', 'zipcode_str', 'month', 'dow'], axis = 1)


# ### c. Re-Checking for multicollinearity: 再マルチコ検出

# In[23]:


df_vif = df_en_fin.drop(["price"], axis=1)


# In[24]:


for cname in df_vif.columns:  
    y=df_vif[cname]
    X=df_vif.drop(cname, axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    rsquared = regr.score(X,y)
    #print(cname,":" ,1/(1-np.power(rsquared,2)))
    if rsquared == 1:
        #print(np.mean((y - regr.predict(X))**2))
        print(cname,X.columns[(regr.coef_> 0.5) | (regr.coef_ < -0.5)])


# In[25]:


df_en_fin = df_en_fin.drop(["sqft_basement","zip_98001","month1","dow1"], axis = 1)


# In[26]:


df_vif = df_en_fin.drop(["price"], axis=1)
for cname in df_vif.columns:  
    y=df_vif[cname]
    X=df_vif.drop(cname, axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    rsquared = regr.score(X,y)
    #print(cname,":" ,1/(1-np.power(rsquared,2)))
    if rsquared == 1:
        print(cname,X.columns[(regr.coef_> 0.5) | (regr.coef_ < -0.5)])


# マルチコの発生を抑えることができることを確認

# ## 8. Re-Training Model: モデルの再構築

# In[27]:


# priceと相関係数が高いTop5+特徴量エンジニアリングで作成したデータ 
y = df_en_fin["price"]
X = df_en_fin.drop(["price","bedrooms","sqft_lot","floors"
         ,"waterfront","view","condition"
         ,"yr_built","lat","long","sqft_lot15"], axis = 1)

regr = LinearRegression(fit_intercept = True)
regr.fit(X, y)

print(X.columns)
print("score = %s"%regr.score(X, y))


# 特徴エンジニアリングなしの場合より、スコアが大きく改善した

# In[28]:


#  データを極力削除しないパターン
y = df_en_fin["price"]
X = df_en_fin.drop(['price'], axis=1)
regr = LinearRegression(fit_intercept = True)
regr.fit(X, y)

print(X.columns)
print("score = %s"%regr.score(X, y))


# 相関関係の強いデータに絞って回帰モデルを作成したものより、  特徴量エンジニアリングの結果を受けてデータの削除は最小限にした方が、良いスコアが出た。

# 学習データとテストデータに分割してモデルを作成する。

# In[29]:


from sklearn.model_selection import train_test_split

# train_test_split関数を使用して、80%学習に、20%テストに当てる
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

regr = LinearRegression(fit_intercept=True)
regr.fit(X_train, y_train)

print("train score = %s"%regr.score(X_train, y_train))


# ## 9. Re-Evalueation: 再評価

# In[30]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_pred = regr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE = %s"%round(mse,3) )
print("RMSE = %s"%round(np.sqrt(mse), 3) )
print("MAE = %s"%round(mae,3) )


# ## 10. Re-Discussion: 再考察
# 当初スコアが低かった要因は、下記2つと考えられる  
# 1) 特徴量エンジニアリング未実施だったこと  
# 2) 相関係数が高い上位のものだけを根拠もなく説明変数として扱ったこと

# In[ ]:




