#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
housing_file_path = '../input/kc_house_data.csv'
housing_data = pd.read_csv(housing_file_path)
#print(housing_data.head())

#to prec=dict the housing price
y=housing_data.price

#the predictors
housing_predictors = ['id','bathrooms','floors','bedrooms','yr_built','yr_renovated','sqft_lot']
x= housing_data[housing_predictors]

#setting the model
housing_model = DecisionTreeRegressor()

#fit the data i.e.., x and y
housing_model.fit(x,y)


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
housing_file_path = '../input/kc_house_data.csv'
housing_data = pd.read_csv(housing_file_path)
#print(housing_data.head())

#to prec=dict the housing price
y=housing_data.price

#the predictors
housing_predictors = ['bathrooms','floors','bedrooms','yr_built','yr_renovated','sqft_lot']
x= housing_data[housing_predictors]

#housing_data.dtypes.sample(10)
#encoded_housing_data = pd.get_dummies(housing_data)

#setting the model
housing_model = DecisionTreeRegressor()
#fit the model
housing_model.fit(x,y)
#print(encoded_housing_data.columns)

print("to make predictions for the following houses")
print((x.head()))

print("the predicted prices are")
print((housing_model.predict(x.head())))


# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

housing_file_path = '../input/kc_house_data.csv'
housing_data = pd.read_csv(housing_file_path)

y = housing_data.price
housing_predictors = ['bathrooms','floors','bedrooms','yr_built','yr_renovated','sqft_lot']
x = housing_data[housing_predictors]

#modeling
housing_model = DecisionTreeRegressor()
housing_model.fit(x,y)

predicted_prices = housing_model.predict(x)
mean_absolute_error(y, predicted_prices)


# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#read the data
housing_file_path = '../input/kc_house_data.csv'
housing_data = pd.read_csv(housing_file_path)

#set the value to be predicted and the predictors
y = housing_data.price
housing_predictors = ['bathrooms','floors','bedrooms','yr_built','yr_renovated','sqft_lot']
x = housing_data[housing_predictors]

#splitting the data into two parts one to train the model and another to validata the model
train_x, val_x, train_y, val_y = train_test_split(x,y,random_state = 0)

#using decision tree model and we fit the training data 
housing_model = DecisionTreeRegressor()
housing_model.fit(train_x, train_y)

#we predict for the validation data and find the mean absolute error 
predicted_values = housing_model.predict(val_x)
mean_absolute_error(val_y,predicted_values)


# In[ ]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#reading the data
housing_file_path = '../input/kc_house_data.csv'
housing_data = pd.read_csv(housing_file_path)

#setting the value to be predicted and the predictors
y = housing_data.price
housing_predictors = ['id','bathrooms','floors','bedrooms','yr_built','yr_renovated','sqft_lot']
x = housing_data[housing_predictors]

#splitting the data into two parts one to train the modl and another to validata the model
train_x,val_x,train_y,val_y = train_test_split(x,y,random_state = 0)

#using random forest model and we fit the training data 
housing_model = RandomForestRegressor()
housing_model.fit(train_x,train_y)

#we predict for the validation data and find the mean absolute error 
predicted_values = housing_model.predict(val_x)
print((mean_absolute_error(val_y, predicted_values)))

#to make the output file
submission_data = pd.DataFrame({'ID':val_x.id,'prices':predicted_values})
submission_data.to_csv('submission.csv',index = False)


# In[ ]:


print((housing_data.isnull().sum()))


# In[ ]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

housing_data = pd.read_csv('../input/kc_house_data.csv')

y = housing_data.price
predictors = ['id','bathrooms','floors','bedrooms','yr_built','yr_renovated','sqft_lot']
x = housing_data[predictors]

train_x, val_x, train_y, val_y = train_test_split(x,y,random_state = 42)
housing_model = RandomForestRegressor()
housing_model.fit(train_x,train_y)

predictions = housing_model.predict(val_x)

#print(mean_absolute_error(val_y, predictions))
print((housing_data.isnull().sum()))

#finding the columns that have missing data
cols_with_missing_data = [col for col in housing_data.columns if housing_data[col].isnull().any()]

#dropping the columns with missing data in housing_data
reduced_housing_data = housing_data.drop(cols_with_missing_data, axis = 1)

#dropping the columns with missing data in tre=aining data
reduced_train_data = housing_data.drop(cols_with_missing_data, axis = 1)

