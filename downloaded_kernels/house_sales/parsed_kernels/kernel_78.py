#!/usr/bin/env python
# coding: utf-8

# # Table of contents
# 1. [Introduction](#introduction)
#     1. [Imports](#imports)
#     2. [Loading data](#load_data)
# 2. [Exploratory Data Analysis](#eda)
#     1. [Data info](#data_info)
#     2. [Price distribution](#price_distr)
#     3. [Feature vs price plots](#feat_vs_price)
#     4. [Correlation matrix](#corr_mat)
# 3. [Data preparation](#data_prep)
#     1. ['33 bedrooms' case](#33bedrm)
#     2. [Outliers handling](#outliers)
#     3. [Visualisations of data without outliers](#expl2)
#     4. [Picking features and creating datasets](#datasets)
#     5. [Data spliting to test and train samples](#split)
# 4. [Machine learning models](#ml_intro)
#     1. [Linear regression](#lr)
#     2. [KNeighbors](#knn)
#     3. [RandomForest regression](#rf)
# 5. [Results overview](#results)
#     1. [R$^{2}$ scores combined](#r_comb)
#     2. [R$^{2}$ vs dataset for each model](#r_vs_data)
# 6. [Conclusions](#concl)

# ## Introduction <a name="introduction"></a>
# Data source and Column Metadata:
# https://www.kaggle.com/harlfoxem/housesalesprediction/data  
# 
# The purpose of this analysis was to practice SciKit-learn and Pandas. 

# #### Imports <a name="imports"></a>

# In[ ]:



import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import eli5


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 12, 8 # universal plot size
pd.options.mode.chained_assignment = None  # default='warn', disables pandas warnings about assigments
njobs = 2 # number of jobs
sbr_c = "#1156bf" # seaborn plot color


# ##### Loading Data <a name='load_data'></a>

# In[ ]:


data = pd.read_csv('../input/kc_house_data.csv', iterator=False, parse_dates=['date'])


# ## Exploratory Data Analysis <a name='eda'></a>

# #### Basic data voerview

# In[ ]:


data.head(10) # to see the columns and first 10 rows


# In[ ]:


data.info() # overview of the data


# Year of pricing distribution. In case of greater variance of data I would consider removing the older records.

# In[ ]:


data['date'].dt.year.hist() 
plt.title('Year of pricing distribution')
plt.show()


# In[ ]:


data.describe() # overview of the data


# #### Price distribution <a name='price_distr' ></a>

# In[ ]:


data['price'].hist(xrot=30, bins=500) 
plt.title('Price distribution')
plt.show()


# #### Feature vs price plots <a name='feat_vs_price'></a>

# In[ ]:


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize = (12, 15))
sns.stripplot(x = "grade", y = "price", data = data, jitter=True, ax = ax1, color=sbr_c)
sns.stripplot(x = "view", y = "price", data = data, jitter=True, ax = ax2, color=sbr_c)
sns.stripplot(x = "bedrooms", y = "price", data = data, jitter=True, ax = ax3, color=sbr_c)
sns.stripplot(x = "bathrooms", y = "price", data = data, jitter=True, ax = ax4, color=sbr_c)
sns.stripplot(x = "condition", y = "price", data = data, jitter=True, ax = ax5, color=sbr_c)
sns.stripplot(x = "floors", y = "price", data = data, jitter=True, ax = ax6, color=sbr_c)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=60)
for i in range(1,7):
    a = eval('ax'+str(i))
    a.set_yscale('log')
plt.tight_layout()


# In[ ]:


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize = (12, 12))
sns.regplot(x = 'sqft_living', y = 'price', data = data, ax = ax1, fit_reg=False, scatter_kws={"s": 1})
sns.regplot(x = 'sqft_lot', y = 'price', data = data, ax = ax2, fit_reg=False, scatter_kws={"s": 1})
sns.regplot(x = 'yr_built', y = 'price', data = data, ax = ax5, fit_reg=False, scatter_kws={"s": 1})
sns.regplot(x = 'sqft_basement', y = 'price', data = data, ax = ax6, fit_reg=False, scatter_kws={"s": 1})
sns.regplot(x = 'lat', y = 'price', data = data, ax = ax3, fit_reg=False, scatter_kws={"s": 1})
sns.regplot(x = 'long', y = 'price', data = data, ax = ax4, fit_reg=False, scatter_kws={"s": 1})
ax6.set_xlim([-100, max(data['sqft_basement'])]) # 6th plot has broken xscale
for i in range(1,7):
    a = eval('ax'+str(i))
    a.set_yscale('log')
plt.tight_layout()


# #### Correlations matrix <a name='corr_mat'></a>

# In[ ]:


corrmat = data.corr() # correlations between features
f, ax = plt.subplots(figsize=(16,16))
sns.heatmap(corrmat, square = True, cmap = 'RdBu_r', vmin = -1, vmax = 1, annot=True, fmt='.2f', ax = ax)


# ## Data preparation <a name='data_prep'></a>

# #### "33 bedrooms" case <a name='33bedrm'></a>
# By taking a look at data.describe() one can see house with 33 bedrooms, which seem to be strange in compare to the others. In the next few lines I will try to examine this case. I guess that's a typo and it should be "3" instead of "33".

# In[ ]:


# selecting house with 33 bedrooms
myCase = data[data['bedrooms']==33]
myCase


# In[ ]:


# data without '33 bedrooms' house
theOthers = data[data['bedrooms']!=33]
theOtherStats = theOthers.describe()
theOtherStats


# In[ ]:


newDf = theOthers[['bedrooms', 'bathrooms', 'sqft_living']]
newDf = newDf[(newDf['bedrooms'] > 0) & (newDf['bathrooms'] > 0)]
newDf['bathrooms/bedrooms'] = newDf['bathrooms']/newDf['bedrooms']
newDf['sqft_living/bedrooms'] = newDf['sqft_living']/newDf['bedrooms']


# In[ ]:


newDf['bathrooms/bedrooms'].hist(bins=20)
plt.title('bathrooms/bedrooms ratio distribution')
plt.show()


# In[ ]:


newDf['sqft_living/bedrooms'].hist(bins=20)
plt.title('sqft_living/bedrooms ratio distribution')
plt.show()


# ##### Bathrooms/Bedrooms ratio

# In[ ]:


# values for other properties
othersMeanBB = np.mean(newDf['bathrooms/bedrooms']) # mean bathroom/bedroom ratio
othersStdBB = np.std(newDf['bathrooms/bedrooms']) # std of bathroom/bedroom ratio

# values for suspicious house: myCase - real data; myCase2 - if there would be 3 bedrooms
myCaseBB = float(myCase['bathrooms'])/float(myCase['bedrooms'])
myCase2BB = float(myCase['bathrooms'])/3. # if there would be 3 bedrooms

print(('{:10}: {:6.3f} bathroom per bedroom'.format('"33" case', myCaseBB)))
print(('{:10}: {:6.3f} bathroom per bedroom'.format('"3" case', myCase2BB)))
print(('{:10}: {:6.3f} (std: {:.3f}) bathroom per bedroom'.format('The others', othersMeanBB, othersStdBB)))


# ##### sqft_living/Bedrooms ratio

# In[ ]:


# values for other properties
othersMeanSB = np.mean(newDf['sqft_living/bedrooms']) # mean sqft_living/bedroom ratio
othersStdSB = np.std(newDf['sqft_living/bedrooms']) # std of sqft_living/bedroom ratio

# values for suspicious house: myCase - real data; myCase2 - if there would be 3 bedrooms
myCaseSB = float(myCase['sqft_living'])/float(myCase['bedrooms'])
myCase2SB = float(myCase['sqft_living'])/3. # if there would be 3 bedrooms

print(('{:10}: {:6.0f} sqft per bedroom'.format('"33" case', myCaseSB)))
print(('{:10}: {:6.0f} sqft per bedroom'.format('"3" case', myCase2SB)))
print(('{:10}: {:6.0f} (std: {:.0f}) sqft per bedroom'.format('The others', othersMeanSB, othersStdSB)))


# ###### Conclusion:
# "House with 33 bedrooms" dosen't look realistic. It will be discarded from the dataset.

# In[ ]:


toDropIndex = myCase.index


# In[ ]:


data.drop(toDropIndex, inplace=True)


# In[ ]:


stats = data.describe()
stats


# #### Outliers handling <a name ='outliers'></a>
# Figures show that there are some outliers in data.
# Data2 is 2nd dataset with arbitrary excluded outliers. Data2 will contain rows that's price do not differ from the mean price by more than 3 std.

# In[ ]:


data2 = data[np.abs(data['price'] - stats['price']['mean']) <= (3*stats['price']['std'])] # cutting 'price'


# #### Visualisations of data without otliers <a name='expl2'></a>

# In[ ]:


data2.describe()


# In[ ]:


sns.regplot(x = "sqft_living", y = "price", data = data2, fit_reg=False, scatter_kws={"s": 2})


# In[ ]:


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize = (12, 15))
sns.stripplot(x = "grade", y = "price", data = data2, jitter=True, ax = ax1, color=sbr_c)
sns.stripplot(x = "view", y = "price", data = data2, jitter=True, ax = ax2, color=sbr_c)
sns.stripplot(x = "bedrooms", y = "price", data = data2, jitter=True, ax = ax3, color=sbr_c)
sns.stripplot(x = "bathrooms", y = "price", data = data2, jitter=True, ax = ax4, color=sbr_c)
sns.stripplot(x = "condition", y = "price", data = data2, jitter=True, ax = ax5, color=sbr_c)
sns.stripplot(x = "floors", y = "price", data = data2, jitter=True, ax = ax6, color=sbr_c)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
for i in range(1,7):
    a = eval('ax'+str(i))
    a.set_yscale('log')
plt.tight_layout()


# In[ ]:


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize = (12, 12))
sns.regplot(x = 'sqft_living', y = 'price', data = data2, ax = ax1, fit_reg=False, scatter_kws={"s": 1})
sns.regplot(x = 'sqft_lot', y = 'price', data = data2, ax = ax2, fit_reg=False, scatter_kws={"s": 1})
sns.regplot(x = 'yr_built', y = 'price', data = data2, ax = ax5, fit_reg=False, scatter_kws={"s": 1})
sns.regplot(x = 'sqft_basement', y = 'price', data = data2, ax = ax6, fit_reg=False, scatter_kws={"s": 1})
sns.regplot(x = 'lat', y = 'price', data = data2, ax = ax3, fit_reg=False, scatter_kws={"s": 1})
sns.regplot(x = 'long', y = 'price', data = data2, ax = ax4, fit_reg=False, scatter_kws={"s": 1})
ax6.set_xlim([-100, max(data2['sqft_basement'])]) # 6th plot has broken xscale
for i in range(1,7):
    a = eval('ax'+str(i))
    a.set_yscale('log')
plt.tight_layout()


# In[ ]:


fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize = (12, 6))
sns.regplot(x = 'sqft_basement', y = 'sqft_living', data = data2, ax = ax1, fit_reg=False, scatter_kws={"s": 1})
sns.regplot(x = 'sqft_above', y = 'sqft_living', data = data2, ax = ax2, fit_reg=False, scatter_kws={"s": 1})
plt.tight_layout()


# #### Picking features and creating datasets <a name='datasets'></a>

# First we should pick features that we will put into model. Correlation matrix presented above might be helpful while making this decision. From listed features I would use:
# * basement
# * bathrooms
# * bedrooms
# * grade
# * sqft_living
# * sqft_lot
# * waterfront
# * view
# 
# 'sqft_basement' and 'sqft_above' seem to be connected with 'sqft_living', so taking into account only 'sqft_living' should work. In case of 'sqft_basement' I will change int value of area to int (0, 1) value indicating whether estate has basement or not.

# In[ ]:


data['basement'] = data['sqft_basement'].apply(lambda x: 1 if x > 0 else 0)
data2['basement'] = data2['sqft_basement'].apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


data2.head(10)


# In[ ]:


# removing unnecessary features
dataRaw = data.copy(deep=True)
dataRaw.drop(['date', 'id'], axis = 1, inplace=True)
dataSel1 = data[['price', 'basement', 'bathrooms', 'bedrooms', 'grade', 'sqft_living', 'sqft_lot', 'waterfront', 'view']]
dataSel2 = data2[['price', 'basement', 'bathrooms', 'bedrooms', 'grade', 'sqft_living', 'sqft_lot', 'waterfront', 'view']]


# #### Data spliting to test and train samples <a name='split'></a>

# In[ ]:


# random_state=seed fixes RNG seed. 80% of data will be used for training, 20% for testing.
seed = 2
splitRatio = 0.2

# data with outliers, only columns selected manually
train, test = train_test_split(dataSel1, test_size=splitRatio, random_state=seed) 
Y_trn1 = train['price'].tolist()
X_trn1 = train.drop(['price'], axis=1)
Y_tst1 = test['price'].tolist()
X_tst1 = test.drop(['price'], axis=1)

# data without outliers, only columns selected manually
train2, test2 = train_test_split(dataSel2, test_size=splitRatio, random_state=seed)
Y_trn2 = train2['price'].tolist()
X_trn2 = train2.drop(['price'], axis=1)
Y_tst2 = test2['price'].tolist()
X_tst2 = test2.drop(['price'], axis=1)

# data with outliers and all meaningful columns (date and id excluded)
trainR, testR = train_test_split(dataRaw, test_size=splitRatio, random_state=seed)
Y_trnR = trainR['price'].tolist()
X_trnR = trainR.drop(['price'], axis=1)
Y_tstR = testR['price'].tolist()
X_tstR = testR.drop(['price'], axis=1)


# In[ ]:


X_trnR.head()


# ## Machine learning models <a name ='ml_intro'></a>

# #### Linear regression <a name='lr'></a>

# In[ ]:


modelLRR = LinearRegression(n_jobs=njobs)
modelLR1 = LinearRegression(n_jobs=njobs)
modelLR2 = LinearRegression(n_jobs=njobs)


# In[ ]:


modelLRR.fit(X_trnR, Y_trnR)
modelLR1.fit(X_trn1, Y_trn1)
modelLR2.fit(X_trn2, Y_trn2)


# In[ ]:


scoreR = modelLRR.score(X_tstR, Y_tstR)
score1 = modelLR1.score(X_tst1, Y_tst1)
score2 = modelLR2.score(X_tst2, Y_tst2)

print(("R^2 score: {:8.4f} for {}".format(scoreR, 'Raw data')))
print(("R^2 score: {:8.4f} for {}".format(score1, 'Dataset 1 (with outliers)')))
print(("R^2 score: {:8.4f} for {}".format(score2, 'Dataset 2 (without outliers)')))


# In[ ]:


lrDict = {'Dataset': ['Raw data', 'Dataset 1', 'Dataset 2'], 
         'R^2 score': [scoreR, score1, score2],
         'Best params': [None, None, None]}
pd.DataFrame(lrDict)


# In[ ]:


lr = LinearRegression(n_jobs=njobs, normalize=True)
lr.fit(X_trnR, Y_trnR)


# Extracting weights of features (not normalized)

# In[ ]:


weights = eli5.explain_weights_df(lr) # weights of LinearRegression model for RawData
rank = [int(i[1:]) for i in weights['feature'].values[1:]]
labels = ['BIAS'] + [X_trnR.columns[i] for i in rank]
weights['feature'] = labels
weights


# #### KNeighbors <a name='knn'></a>

# KNeighbors Regressor requires more parameters than Linear Regression, so using GridSearchCV to tune hyperparameters seem to be good idea.

# In[ ]:


tuned_parameters = {'n_neighbors': list(range(1,21)), 'weights': ['uniform', 'distance']}
knR = GridSearchCV(KNeighborsRegressor(), tuned_parameters, n_jobs=njobs)
kn1 = GridSearchCV(KNeighborsRegressor(), tuned_parameters, n_jobs=njobs)
kn2 = GridSearchCV(KNeighborsRegressor(), tuned_parameters, n_jobs=njobs)


# In[ ]:


knR.fit(X_trnR, Y_trnR)
kn1.fit(X_trn1, Y_trn1)
kn2.fit(X_trn2, Y_trn2)


# In[ ]:


scoreR = knR.score(X_tstR, Y_tstR)
score1 = kn1.score(X_tst1, Y_tst1)
score2 = kn2.score(X_tst2, Y_tst2)
parR = knR.best_params_
par1 = kn1.best_params_
par2 = kn2.best_params_

print(("R^2: {:6.4f} {:12} | Params: {}".format(scoreR, 'Raw data', parR)))
print(("R^2: {:6.4f} {:12} | Params: {}".format(score1, 'Dataset 1', par1)))
print(("R^2: {:6.4f} {:12} | Params: {}".format(score2, 'Dataset 2', par2)))


# In[ ]:


knDict = {'Dataset': ['Raw data', 'Dataset 1', 'Dataset 2'], 
         'R^2 score': [scoreR, score1, score2],
         'Best params': [parR, par1, par2]}
pd.DataFrame(knDict)


# #### RandomForest regression <a name='rf'></a>

# As in the previous case using GridSearchCV will help with tunning hyperparameters.

# In[ ]:


tuned_parameters = {'n_estimators': [10,20,50,100], 'max_depth': [10,20,50]}
rfR = GridSearchCV(RandomForestRegressor(), tuned_parameters, n_jobs=njobs)
rf1 = GridSearchCV(RandomForestRegressor(), tuned_parameters, n_jobs=njobs)
rf2 = GridSearchCV(RandomForestRegressor(), tuned_parameters, n_jobs=njobs)


# In[ ]:


rfR.fit(X_trnR, Y_trnR)
rf1.fit(X_trn1, Y_trn1)
rf2.fit(X_trn2, Y_trn2)


# In[ ]:


scoreR = rfR.score(X_tstR, Y_tstR)
score1 = rf1.score(X_tst1, Y_tst1)
score2 = rf2.score(X_tst2, Y_tst2)
parR = rfR.best_params_
par1 = rf1.best_params_
par2 = rf2.best_params_

print(("R^2: {:6.4f} {:12} | Params: {}".format(scoreR, 'Raw data', parR)))
print(("R^2: {:6.4f} {:12} | Params: {}".format(score1, 'Dataset 1', par1)))
print(("R^2: {:6.4f} {:12} | Params: {}".format(score2, 'Dataset 2', par2)))


# In[ ]:


rfDict = {'Dataset': ['Raw data', 'Dataset 1', 'Dataset 2'], 
         'R^2 score': [scoreR, score1, score2],
         'Best params': [parR, par1, par2]}
pd.DataFrame(rfDict)


# Checking feature importances in Random Forest Regressor model.

# In[ ]:


rf = RandomForestRegressor(n_estimators=100, max_depth=50, n_jobs=njobs)


# In[ ]:


rf.fit(X_trnR, Y_trnR)


# In[ ]:


importances = rf.feature_importances_

# calculating std by collecting 'feature_importances_' from every tree in forest
rfStd = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
indices = np.argsort(importances)[::-1] # descending order


# In[ ]:


xlabels = [X_trnR.columns[i] for i in indices]


# In[ ]:


plt.title("Random Forest: Mean feature importances with STD")
plt.bar(list(range(len(xlabels))), importances[indices],
       color="#1156bf", yerr=rfStd[indices], align="center", capsize=8)
plt.xticks(rotation=45)
plt.xticks(list(range(len(xlabels))), xlabels)
plt.xlim([-1, len(xlabels)])
plt.show()


# In[ ]:


# feature importance for RandomForest with the best params tunnedy by GridSearchCV calculated by eli5
weights = eli5.explain_weights_df(rf) 
rank = [int(i[1:]) for i in weights['feature'].values]
labels = [X_trnR.columns[i] for i in rank]
weights['feature'] = labels
weights


# ## Results overview <a name='results'></a>
# lr: LinearRegression  
# kn: KNeighborsRegressor   
# rf: RandomForestRegressor  

# In[ ]:


resDict = {'lr' : lrDict, 'kn' : knDict, 'rf' : rfDict}


# In[ ]:


dict_of_df = {k: pd.DataFrame(v) for k,v in list(resDict.items())}
resDf = pd.concat(dict_of_df, axis=0)
resDf


# ##### R$^{2}$ scores combined <a name='r_comb'></a>

# In[ ]:


toPlot = resDf.sort_values(by=['R^2 score'], ascending=False)
fig, axes = plt.subplots(ncols=1, figsize=(12, 8))
toPlot['R^2 score'].plot(ax=axes, kind='bar', title='R$^{2}$ score', color="#1153ff")
plt.ylabel('R$^{2}$', fontsize=20)
plt.xlabel('Model & Dataset', fontsize=20)
plt.xticks(rotation=45)
plt.show()


# ##### R$^{2}$ vs dataset for each model <a name='r_vs_data'></a>

# In[ ]:


toPlot = resDf.sort_values(by=['R^2 score'], ascending=False)
fig, axes = plt.subplots(ncols=1, figsize=(12, 8))
toPlot.loc['lr']['R^2 score'].plot(ax=axes, kind='bar', title='R$^{2}$ score for Linear Regression', color="#1153ff")
plt.ylabel('R$^{2}$', fontsize=20)
plt.xlabel('Dataset', fontsize=20)
plt.xticks(rotation=45)
plt.xticks(list(range(3)), [toPlot.loc['lr']['Dataset'][i] for i in range(3)])
plt.show()


# In[ ]:


toPlot = resDf.sort_values(by=['R^2 score'], ascending=False)
fig, axes = plt.subplots(ncols=1, figsize=(12, 8))
toPlot.loc['kn']['R^2 score'].plot(ax=axes, kind='bar', title='R$^{2}$ score for KNeighbors', color="#1153ff")
plt.ylabel('R$^{2}$', fontsize=20)
plt.xlabel('Dataset', fontsize=20)
plt.xticks(rotation=45)
plt.xticks(list(range(3)), [toPlot.loc['kn']['Dataset'][i] for i in range(3)])
plt.show()


# In[ ]:


toPlot = resDf.sort_values(by=['R^2 score'], ascending=False)
fig, axes = plt.subplots(ncols=1, figsize=(12, 8))
toPlot.loc['rf']['R^2 score'].plot(ax=axes, kind='bar', title='R$^{2}$ score for Random Forest', color="#1153ff")
plt.ylabel('R$^{2}$', fontsize=20)
plt.xlabel('Dataset', fontsize=20)
plt.xticks(rotation=45)
plt.xticks(list(range(3)), [toPlot.loc['rf']['Dataset'][i] for i in range(3)])
plt.show()


# ## Conclusions <a name='concl'></a>

# Random Forest appeared to be the best model among tested models. Raw Data turned out to be the most (independently for model used) reliable.

# In[ ]:




