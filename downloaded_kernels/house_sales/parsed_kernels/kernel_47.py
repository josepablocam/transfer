#!/usr/bin/env python
# coding: utf-8

# <h2>Introduction:</h2>
# This is my First kernel, I have attempted to understand which features contribute to the Price of the houses.
# <br> A shoutout to SRK and Anisotropic from whom iv learned a lot about data visualisation</br>

# <h2>Lets import the libraries we need for now<h2>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# 
# <h2>now we import the dataset</h2>
# 

# In[ ]:


data = pd.read_csv('../input/kc_house_data.csv')


# In[ ]:


# Lets check it out 
data.head()


# <h2>now lets check out how many NaN values are there</h2>

# In[ ]:


data.isnull().sum()


# wow! so we just dont need to bother about using Imputer and handeling the NaN values
# <br>Now lets check out how the data actually is</br>

# In[ ]:


print((data.info()))
print(("**"*40))
print((data.describe()))


# Oops, we forgot to convert the date to datetime, lets get that done first

# In[ ]:


data['date'] = pd.to_datetime(data['date'])
# while im at it, let me create a year and month column too
data['year'], data['month'] = data['date'].dt.year, data['date'].dt.month


# In[ ]:


# as we have everything from the date column, lets simply remove it 
del data['date']


# <h2>We have finished the preliminary data cleaning, now lets visualize and check out for some correlation that we can use</h2>

# the dataset includes latitude and longitude for each entry, lets plot it out and see if specific areas sold more houses or less

# In[ ]:


plt.figure(figsize=(12,12))
sns.jointplot( 'long','lat',data = data, size=9 , kind = "hex")
plt.xlabel('Longitude', fontsize=10)
plt.ylabel('Latitude', fontsize=10)
plt.show()


# as we guessed, there are some areas where many houses were sold
# <br>
# <h2>lets try out the pearson correlation</h2></br>
# 

# In[ ]:


dataa = [
    go.Heatmap(
        z= data.corr().values,
        x= data.columns.values,
        y= data.columns.values,
        colorscale='Viridis',
        text = True ,
        opacity = 1.0
        
    )
]

layout = go.Layout(
    title='Pearson Correlation',
    xaxis = dict(ticks='', nticks=30),
    yaxis = dict(ticks='' ),
    width = 800, height = 600,
    
)

fig = go.Figure(data=dataa, layout=layout)
py.iplot(fig, filename='Housedatacorr')


# In the price col we can see some rows are so very close to zero, but lets not remove them from the data set as of yet, it may be useful for some ensemble process
# <br>
# <h2>Now lets try out some trees and ensemble methods for a better understanding of the feature importances</h2></br>

# In[ ]:


# the models we will run
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor

# some metrics to help us out
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as mse


# lets remove the price col from data as we will need it now 

# In[ ]:


target = data['price']
# we dont need the price column in data anymore
del data['price']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)


# Now im going to find feature_importances using various ensemble methods 

# In[ ]:


dr = DecisionTreeRegressor()
dr.fit(X_train,y_train)
drimp = dr.feature_importances_


# In[ ]:


rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(X_train,y_train)
rfrimp = rfr.feature_importances_


# In[ ]:


gbr =  GradientBoostingRegressor(n_estimators=100)
gbr.fit(X_train,y_train)
gbrimp = gbr.feature_importances_


# In[ ]:


abr =  AdaBoostRegressor(n_estimators=100)
abr.fit(X_train,y_train)
abrimp = abr.feature_importances_


# In[ ]:


etr =  ExtraTreesRegressor(n_estimators=100)
etr.fit(X_train,y_train)
etrimp = etr.feature_importances_


# lets create a data frame that has all these values 
# 

# In[ ]:


d = {'Decision Tree':drimp, 'Random Forest':rfrimp, 'Gradient Boost':gbrimp,'Ada boost':abrimp, 'Extra Tree':etrimp}


# In[ ]:


features = pd.DataFrame(data = d)
# lets check out features
features.head()


# One good way to check how important a feature is will be to calculate the mean from each method 

# In[ ]:


features['mean'] = features.mean(axis= 1) 
# we forgot to add the names of the features
features['names'] = data.columns.values


# In[ ]:


#lets check it out now 
features.head()


# <h2>Now i'll plot a barplot to illustrate how the mean of each feature has fared</h2>
# 

# In[ ]:


y = features['mean'].values
x = features['names'].values
data = [go.Bar(
            x= x,
             y= y,
            width = 0.5,
            marker=dict(
               color = features['mean'].values,
            colorscale='Portland',
            showscale=True,
            reversescale = False
            ),
            opacity=0.6
        )]

layout= go.Layout(
    autosize= True,
    title= 'Mean Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance for Housing Price',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='barplothouse')


# <h1>Conclusion</h1>
# <br>We can see that there are two very prominent features i.e. sqft_living and grade, which according to all the models are very useful to predict the price of the house.</br>
# <br>A close third is the Latitude, which one can consider as the area where the house is</br>
# <br> We were also expecting the bathrooms feature and the sqft_above feature to be high ranking in the barplot, as it was considered imp according to Pearson corr. </br>
# <br>But maybe the ensemble's know better or it has simply overfitted the data</br>
# <br> I think there might be a few more features that can be extracted from the dataset which might give more insight , in accordance to our intuitive and qualitative thinking</br>
# <br>Hope you enjoyed it! please give me some feedback below and upvote if you liked it! </br>
