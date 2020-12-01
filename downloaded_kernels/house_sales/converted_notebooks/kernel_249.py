#!/usr/bin/env python
# coding: utf-8

# # 中間報告

# 作成：松島亮輔

# 選んだ課題【House Sales in King County, USA】住宅販売価格を予測する
# 
# 【URL】 https://www.kaggle.com/harlfoxem/housesalesprediction/data

# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
#グラフをnotebook内に描画させるための設定
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.decomposition import PCA #主成分分析用ライブラリ
from sklearn.metrics import mean_squared_error, mean_absolute_error
from IPython.display import display
import seaborn as sns
from scipy.stats import norm
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression #線形回帰のライブラリ
from sklearn.linear_model import Ridge,Lasso,ElasticNet #正則化項付き最小二乗法を行うためのライブラリ
from sklearn.preprocessing import PolynomialFeatures
import math


# In[4]:


df_data = pd.read_csv("../input/kc_house_data.csv")
df_data["price"] = df_data["price"] / 10**6 #単位を100万ドルにしておく
print(df_data.columns)
display(df_data.head())
display(df_data.tail())


# In[5]:


pd.plotting.scatter_matrix(df_data, figsize=(30,30))
plt.show()


# # 有効と考えられる説明変数

# 現段階で有効と考えられる説明変数は各変数の関係を示すグラフよりbedrooms,bahtrooms,sqft_living,,sqft_above,sqft_basement,sqft_living15と考える。 またこれらの説明変数とpriceとの関係は一次の線形近似ができるが説明変数の値が大きくなるにつれて乱れのが大きくなっていく。各変数の関係を示すグラフより、これらの説明変数とpriceとの関係は全てこの関係に当てはまっており異常値ないと考えた。

# # bathroomsとpriceの関係に注目して分析

# In[8]:


df_data.plot.scatter(x='bathrooms', y='price',figsize=(10,10))


# In[9]:


sns.jointplot(x="bathrooms", y="price", data=df_data,kind = 'reg', size = 10)
plt.show()


# # 主成分分析

# In[10]:


X = np.array(df_data[["bathrooms","price"]])

pca = PCA(n_components=2) #主成分分析用のオブジェクトをつくる。削減後の次元数を引数で指定する。2次元データなので、3以上にするとエラーになる
pca.fit(X) #主成分分析の実行
#Y = np.dot((X - pca.mean_), pca.components_.T)
Y = np.dot((X), pca.components_.T)

dataframe_value = pd.DataFrame(Y)
dataframe_value.columns = ['a', 'b']

sns.jointplot(x="a", y="b", data=dataframe_value,kind = 'reg', size = 10)
plt.show()


# # 縦軸の分布が正規分布に似ている

# In[11]:


opp = dataframe_value.assign(
    round_A=lambda dataframe_value: dataframe_value.a.round(), # 四捨五入
    )
sns.jointplot(x="round_A", y="b", data=opp,kind = 'reg', size = 10)
plt.show()


# In[12]:


df = opp[opp.round_A == 2]
param = norm.fit(df['b'])
print(param)
x = np.linspace(-2,3,100)
pdf_fitted = norm.pdf(x,loc=param[0], scale=param[1])
pdf = norm.pdf(x)
plt.figure
plt.title('Normal distribution')
plt.plot(x, pdf_fitted, 'r-')
plt.hist(df['b'], normed=1, alpha=.3)
plt.show()


# In[13]:


n = 2

x = []
df_param_loc = []
df_param_scale= []

while n != 6:
    df = opp[opp.round_A == n]
    param = norm.fit(df['b'])
    print(param)
    x += [n]
    df_param_loc += [param[0]]
    df_param_scale+= [param[1]]
    
    n += 1
    


# In[14]:


plt.scatter(x,df_param_loc)


# In[15]:


plt.scatter(x,df_param_scale)


# In[16]:


x = np.array(x)
X = x.reshape(-1,1)
y = df_param_loc

regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X, y)

a0 = regr.intercept_
a1 = regr.coef_

plt.plot(x, y, 'o')
plt.plot(x, a0+a1*x)
plt.show()


# In[17]:


x = np.array(x)
X = x.reshape(-1,1)
y = df_param_scale

regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X, y)

b0 = regr.intercept_
b1 = regr.coef_

plt.plot(x, y, 'o')
plt.plot(x, b0+b1*x)
plt.show()


# $μ(x)=a0L+a1*x$
# 
# $σ(x)=b0+b1*x$
# 
# $f(x)=\frac{1}{\sqrt{2πσ^2}}exp(\frac{−(y−μ)^2}{2σ^2})$

# In[22]:


bathrooms_num = 4

df_test = pd.DataFrame({"price":np.linspace(0,6,100),
                        "bathrooms":bathrooms_num})
price = np.linspace(0,6,601)

X = np.array(df_test[["bathrooms","price"]])

Y = np.dot(X, pca.components_.T)

dataframe_value = pd.DataFrame(Y)
dataframe_value.columns = ['a', 'b']

x = np.array(dataframe_value["a"])
y = np.array(dataframe_value["b"])

my = a0 + a1*x
sig = b0 + b1*x

norm = (np.exp(-(y-my)**2/(2*sig**2))) / np.sqrt(2*math.pi*sig**2)

plt.figure
plt.plot(np.linspace(0,6,100),norm, 'r-')
plt.show()


# In[ ]:




