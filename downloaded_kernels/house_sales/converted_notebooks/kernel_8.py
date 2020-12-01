#!/usr/bin/env python
# coding: utf-8

# # Linear Regression预测房价

# In[5]:


import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型
from sklearn.model_selection import train_test_split # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
import math #数学库


# 从../input/kc_house_data.csv文件中读入数据

# In[6]:


data = pd.read_csv("../input/kc_house_data.csv")
data


# In[7]:


data.dtypes


# 获得自变量X和因变量Y

# In[8]:


X = data[['bedrooms','bathrooms','sqft_living','floors']]
Y = data['price']


# 获得2:1的训练：测试数据比例

# In[19]:


xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=1/3, random_state=0)


# In[20]:


xtrain = np.asmatrix(xtrain)
xtest = np.asmatrix(xtest)
ytrain = np.ravel(ytrain)
ytest = np.ravel(ytest)


# 观察房价和生活面积的关系

# In[21]:


plt.scatter(X['sqft_living'], Y)
plt.show()


# 观察生活面积分布

# In[22]:


X['sqft_living'].hist()
plt.show()


# 用xtrain和ytrain训练模型

# In[23]:


model = LinearRegression()
model.fit(xtrain, ytrain)


# In[24]:


pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))


# In[25]:


model.intercept_


# In[26]:


#一个房子，3个卧室，2个卫生间，2500sqft，2层楼
#放入bias 1
model.predict([[3,2,2500,2]])


# 训练集上的均方差MSE

# In[27]:


pred = model.predict(xtrain)
((pred-ytrain)*(pred-ytrain)).sum() / len(ytrain)


# 平均相对误差

# In[28]:


(abs(pred-ytrain)/ytrain).sum() / len(ytrain)


# 训练集合上的MSE

# In[29]:


predtest = model.predict(xtest)
((predtest-ytest)*(predtest-ytest)).sum() / len(ytest)


# In[30]:


(abs(predtest-ytest)/ytest).sum() / len(ytest)


# In[ ]:





# In[ ]:




