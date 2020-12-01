#!/usr/bin/env python
# coding: utf-8

# ### Nov 2017

# This is my first kernel on Kaggle. I picked this housing price dataset since i feel there is so much thing i can do with it. In this kernel, i will focus on data visualization, exploration, and prediction using XGBoost. Please feel free to share any commonts or suggestions. I have learnt so much here during the last few weeks, would really want to learn more through your comments. 

# The first part will be **data visualization and exploration.** Then the second part will be **XGBoost modeling**. Let's get started.

# # Visualization and Exploration

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.colors import Normalize
import seaborn as sns
import operator
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import warnings
warnings.filterwarnings("ignore")


# In[2]:


train = pd.read_csv('../input/kc_house_data.csv',index_col='date',parse_dates=True)
train.info()


# There are no missing values in this dataset. Let's first check the distribution of **housing price**.

# In[3]:


plt.figure(figsize=(8,8))
plt.hist(train.price,bins=100,color='b')
plt.title('Histogram of House Price')
plt.show()
# 


# it's extremely skewed, we may consider to take the log transformation of it.

# In[4]:


train['log_price'] = np.log(train['price'])


# Next we will check the** seasonality of housing price**. In the following plot the y axis is representing the housing price on log scale, the color of the bubbles shows the number of sales. When the number of sales is high, the housing price is also high, which is quite reasonable. But sometimes, the market might response a bit slow, that's why we can see in May of 2015, even though there are not many of sales, the price is still high.
# We can see there are clear seasonality, so create some month indicators might be a good idea.
# 

# In[5]:


# monthly change of prices
train['ym'] = (train.index.year *100 + train.index.month).astype(str) 
ym_summary = train.groupby('ym')['price'].agg(['mean','count'])

vmin = np.min(ym_summary['count'])
vmax = np.max(ym_summary['count'])
norm = colors.Normalize(vmin,vmax)

plt.figure(figsize=(15,7))
plt.scatter(x=np.arange(ym_summary.shape[0]), y =ym_summary['mean'],c= ym_summary['count'],
            s= ym_summary['count'],norm=norm ,alpha = 0.8, cmap='jet')

plt.plot(np.arange(ym_summary.shape[0]), ym_summary['mean'] ,'--')
plt.xticks(np.arange(ym_summary.shape[0]),ym_summary.index.values)
plt.yscale('log')
plt.xlabel('Year-Month')
plt.ylabel('Price (log scale)')
clb = plt.colorbar() 
clb.ax.set_title('number of sales')
plt.title('Averge Price by Month')
plt.show()


# Another very important factor would be **location**. So let's check the housing price by latitude and longitude. As we can see from this graph, the northern area are generally more expensive than the southern. Most of the million-dollar houses (right stars in the graph) are around the hollow area in the north. That should be Lake Washington. Latitude and Longitude can be very good features if we want to build a model to predict the housing price. And they are on a more granular level than zip code. So it should provide more information than zip code. Another point to note from the graph is that the relationship between latitude/longitude and housing price is not linear, so linear model might not work very well. But tree based models should work better in this case. 

# In[6]:


plt.figure(figsize=(15,10))
vmin = np.min(train.price)
vmax = np.max(train.price)
norm = colors.LogNorm(vmin*2,vmax/3)
plt.scatter(train.long,train.lat, marker='*',c=train.price,norm=norm,cmap='jet') 
plt.xlabel('Longitude')
plt.ylabel('Latituede')
plt.title('House Price by Geography')
clb = plt.colorbar() 
clb.ax.set_title('Price')


# Now let's dig into other features. There are couple of variables related to **the house area**. Some of them have very strong correlation. Boosted tree models would intrinsincally do the variable selection, but if we want to build a statistical model, which will be my next task after this kernel, we'd better remove the highly correlated variables. Of course one could also do a PCA etc if you want to extract a single variable from them. Here I will simply take sqft_living15, sqft_above, sqft_lot_15 out.
# 

# In[7]:


corr = train[['sqft_living','sqft_living15','sqft_lot','sqft_lot15','sqft_above','sqft_basement','floors','log_price']].corr()
mask = np.zeros_like(corr,dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f,ax = plt.subplots(figsize=(8,8))
cmap=sns.diverging_palette(120,10,as_cmap=True) #
sns.heatmap(corr,mask=mask,cmap=cmap,center=0,square=False,linewidths=.5,cbar_kws={"shrink":.5})


# Let's check the **bivariate relationship between these area variables with the housing price**. We first make the plot on the original scales.

# In[8]:


def scatter_p(fea,price):
    n_f = len(fea)
    n_row = n_f//3+1
    fig=plt.figure(figsize=(20,15))
    i = 1
    for f in fea:
        x=train[f]
        y=train[price]
        m, b = np.polyfit(x, y, 1)    
        
        ax=fig.add_subplot(n_row,3,i)
        plt.plot(x,y,'.',color='b')
        plt.plot(x, m*x + b, '-',color='r')
        plt.xlabel(f)
        plt.ylabel(price)
        i += 1


# In[9]:


scatter_p(fea=['sqft_living', 'sqft_lot','sqft_basement'],price='price')


# Like we mentioned earlier, there is too much skewness, not only in the housing price, but also in the features. It's better for us to transform them to the log scale. Now the relationship is much clearer and predictable. In the following analysis, we will use log scale for price and sqft variables. 

# In[10]:


train['log_sqft_living'] = np.log(train['sqft_living'])
train['log_sqft_lot'] = np.log(train['sqft_lot'])
train['log_sqft_basement'] = np.log1p(train['sqft_basement'])
scatter_p(fea=['log_sqft_living','log_sqft_lot','log_sqft_basement'],price='log_price')


# From the above plots, we see that **log_sqrt_basement** is the one we need to take some extra care. If ignoring the 0 values, the relationship is quite clear. But the data with 0 values forced the regression line to be much flatter. It might be worth to create an indicator variable for log_sqrt_basement. In linear regression, this would be like give a separate intercept term for the non-zero points.
# 

# In[11]:


train['basement_ind'] = [1 if x>0 else 0 for x in train.sqft_basement]


# Next is another set of variables. 

# In[12]:


scatter_p(fea=['yr_built','yr_renovated','bathrooms'],price='log_price')


# we see similar problems with **yr_renovated**. So we will create another indicator for this variable.

# In[13]:


train['renovated_ind'] = [1 if x>0 else 0 for x in train.yr_renovated]


# In[14]:


x=train.loc[train.renovated_ind==1,'yr_renovated']
y=train.loc[train.renovated_ind==1 ,'log_price']
m, b = np.polyfit(x, y, 1)   
plt.plot(x,y,'.',color='b')
plt.plot(x, m*x + b, '-',color='r')
plt.title('Renovated Houses')
plt.xlabel('yr_renovated')
plt.ylabel('log_price')
plt.show()


# Let's check the correlation for the following set of variables. The collinearity is not as bad as that for the sqft variables. We will keep them in the model.

# In[15]:


corr = train[['bedrooms','condition','grade','view','floors','log_price']].corr()
mask = np.zeros_like(corr,dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f,ax = plt.subplots(figsize=(8,8))
cmap=sns.diverging_palette(120,10,as_cmap=True) #
sns.heatmap(corr,mask=mask,cmap=cmap,center=0,square=False,linewidths=.5,cbar_kws={"shrink":.5})


# Let's see the **bivariate relationship of these variables with housing price**. 
# 

# In[16]:


fig = plt.figure(figsize=(15,12))
fig.add_subplot(3,2,1)
sns.violinplot(x="bedrooms", y="log_price", data=train)    
fig.add_subplot(3,2,2)
sns.violinplot(x="condition", y="log_price", data=train)  
fig.add_subplot(3,2,3)
sns.violinplot(x="grade", y="log_price", data=train)   
fig.add_subplot(3,2,4)
sns.violinplot(x="view", y="log_price", data=train)  
fig.add_subplot(3,2,5)
sns.violinplot(x="floors", y="log_price", data=train)  


# For condition, grad, and view, the relationship is monotone. But for bedrooms and floors, it is not. So in the tree based model, i will treat bedrooms and floors as categorical variable, but the others as continuous.

# In[17]:


# convert bedrooms, floors, year_month to binary feature
train = pd.get_dummies(train,columns=['bedrooms','floors','ym'],drop_first=True)


# **Waterfront** is a binary variable, we can see the distribtution of housing pric is quite different for the two groups. So it is likely to be a good predictor.

# In[18]:


wf = train.waterfront.unique()
for i in wf:
    temp_x=train.loc[train.waterfront==i,'log_price']
    ax = sns.kdeplot(temp_x,shade=True)
plt.title('Distribution of log_price by waterfront')


# # XGBoost

# Now comes to the modeling part. I will build a xgboost model to predict the housing price with the features we selected and created. 

# In[19]:


trn = train.drop(['id','price','log_price','sqft_living15','sqft_lot15','sqft_living','sqft_lot','sqft_above',
                  'sqft_basement','zipcode'],axis=1)
resp = train['log_price']


# In[20]:


from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
from sklearn.metrics import mean_squared_error
import gc
gc.collect()


# In[21]:


X_trn, X_tst, y_trn, y_tst = train_test_split(trn,resp,test_size=0.2,random_state=123)


# In[22]:


param={
    'objective': 'reg:linear',
    'eta'      :0.02,
    'eval_metric':'rmse',
    'max_depth': 5,
    'min_child_weight':3,
    'subsample' : 0.8,
    'colsample_bytree' : 0.8,
    'silent': 1,
    'seed' : 123
}
trn = xgb.DMatrix(X_trn,label=y_trn)
tst = xgb.DMatrix(X_tst,label=y_tst)
res = xgb.cv(param,trn,nfold=4,num_boost_round=2000,early_stopping_rounds=50,
             stratified=True,show_stdv=True,metrics={'rmse'},maximize=False)
min_index=np.argmin(res['test-rmse-mean'])

model = xgb.train(param,trn,min_index,[(trn,'train'),(tst,'test')])
pred = model.predict(tst)
print(('Test RMSE:', np.sqrt(mean_squared_error(y_tst,pred))))


# Let's see how the model fit.

# In[23]:


plt.scatter(y_tst,pred,color='b')
plt.xlabel('true log_price')
plt.ylabel('predicted log_price')


# In[24]:


r_sq = ((pred-np.mean(y_tst))**2).sum() / ((y_tst - np.mean(y_tst))**2).sum()
print(('R square is: ', r_sq))


# Plot the top 20 features according to their importance

# In[25]:


fig,ax = plt.subplots(1,1,figsize=(10,10))
xgb.plot_importance(model,ax=ax,max_num_features=20)


# ### Partial Dependence Plot

# The information in this post [https://xiaoxiaowang87.github.io/monotonicity_constraint/](http://) helped me a lot on making the partial dependence plot. 

# In[26]:


def partial_dependency(bst, feature):
    X_temp = X_trn.copy()
    grid = np.linspace(np.percentile(X_temp.loc[:, feature], 0.1),np.percentile(X_temp.loc[:, feature], 99.5),50)
    y_pred = np.zeros(len(grid))
    for i, val in enumerate(grid):
            X_temp.loc[:, feature] = val
            data = xgb.DMatrix(X_temp)
            y_pred[i] = np.average(bst.predict(data))
    
    plt.plot(grid,y_pred,'-',color='r')
    plt.plot(X_trn.loc[:,feature], y_trn, 'o',color='grey',alpha=0.01)
    plt.title(('Partial Dependence of '+feature))
    plt.xlabel(feature)
    plt.ylabel('Housing Price (log scale)')
    plt.show()
      


# In[27]:


partial_dependency(model,'log_sqft_living')


# In[28]:


partial_dependency(model,'lat')


# In[29]:


partial_dependency(model,'long')


# That would be what i have so far for this kernel. Now there is a question: what if i really want to know how the price varies across different ZIP CODE after controling other factors? If you say "well, the houses at latitude 47.65 and longitude -122.2 are much more expensive than those at latitude 47.3 and longitude -122.2" in the chat with your friends, they probably will be staring at you and wondering whether you are from Mars. So not for the purpose of prediction, but for the sake of interpretability, how can we answer that question? 
# 
# That will be what I am aiming at in my next kernel. 
# 
# Thank you for reading! Again please don't hesitate to leave commonts and suggestions.
