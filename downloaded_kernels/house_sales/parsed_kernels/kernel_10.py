#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This data is 19 house features plus the price, including 21613 observations.
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

house = pd.read_csv("../input/kc_house_data.csv")


# In[ ]:


house.head()


# In[ ]:


house.info()


# In[ ]:


y = house["price"]
X = house.loc[:,"bedrooms":"sqft_lot15"]


# In[ ]:


y.shape, X.shape


# ## Seperate test and trail data

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
X_train.shape, y_train.shape


# ## Fit Model

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)


# ## Cross-validation

# In[ ]:


from sklearn.model_selection import cross_val_score
model = LinearRegression()
scores = cross_val_score(model, X, y, cv=5)
scores


# In[ ]:


np.mean(scores)


# ## Ridge

# In[ ]:


from sklearn.linear_model import Ridge
model = Ridge(alpha=0.1, normalize=True)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[ ]:


coef = model.coef_
coef


# In[ ]:


col_names = house.drop(["id", "date", "price"], axis=1).columns
plt.figure(figsize=(10,10))
_ = plt.plot(list(range(len(col_names))), coef)
_ = plt.xticks(list(range(len(col_names))), col_names, rotation=60)
_ = plt.ylabel("Coefficients")
plt.show()

