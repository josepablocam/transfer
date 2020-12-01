#!/usr/bin/env python
# coding: utf-8

# Step 1:
# Creating data_frame named housing_data

# In[ ]:


import pandas as pd
housing_data = pd.read_csv("../input/kc_house_data.csv")
housing_data.head()


# Step 2:
# 
# Calculating age of house for better analysis
# 
# Creating another column named age_of_house for visualization
# 
# 

# In[ ]:


import datetime
current_year = datetime.datetime.now().year
housing_data["age_of_house"] = current_year - pd.to_datetime(housing_data["date"]).dt.year
housing_data.head()


# Data Frame Info. (Quick View)

# In[ ]:


housing_data.info()


# Populating Column Names

# In[ ]:


housing_data.columns


# Step 3:
# Selecting features and target

# In[ ]:


feature_cols = [ 'age_of_house',  'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
x = housing_data[feature_cols]
y = housing_data["price"]


# Visualizing Feature Columns against target

# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.pairplot(housing_data,x_vars=feature_cols,y_vars="price",size=7,aspect=0.7,kind = 'reg')


# Step 4:
# Splitting Training and Test Data

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=3)


# Step 5:
# Fitting Data to Linear Regressor using scikit

# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# Achieved Accuracy: 66% 
# which is not so bad at inital commit :)

# In[ ]:


accuracy = regressor.score(x_test, y_test)
"Accuracy: {}%".format(int(round(accuracy * 100)))

