#!/usr/bin/env python
# coding: utf-8

# # House Sales in King County, USA
# https://www.kaggle.com/harlfoxem/housesalesprediction/data
# 
# Miki Katsuragi

# In[1]:


import pandas as pd
import numpy as np
from IPython.display import display
from dateutil.parser import parse
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy.stats import norm, skew #for some statistics
from scipy import stats


# ### Load data into Pandas data frame

# In[2]:


d = pd.read_csv("../input/kc_house_data.csv")
d.head()


# ## Initial data cleansing

# In[3]:


#Convert datetime to year and month
d['yr_sold'] = pd.to_datetime(d.date).map(lambda x:x.year)
d['m_sold'] = pd.to_datetime(d.date).map(lambda x:x.month)

#仮説:リノベがあったかなかったかのほうが大事そう
d['yr_renovated'] = np.array(d['yr_renovated'] != 0)*1

#仮設：yr_builtは築年数に直したほうがよさそう
d['yr_built']=d['yr_built'].max()-d['yr_built']

# yr_renovated should be converted to binary values since there are too many 0 and year has almost no numeric meaning
d['yr_renovated'] = np.array(d['yr_renovated'] != 0)*1

#zipcodeを前4桁だけにする
#d['zipcode']=d['zipcode'].astype(str).map(lambda x:x[0:4])
#id is not definitely required for this analysis
#d = d.drop(["id","date","lat","long"], axis=1)

#zipcodeのかわりにlat,long使う
d = d.drop(["id","date","zipcode"], axis=1)


# ### Confirm relationships between price and other parameters¶

# In[4]:


# Draw scatter charts
df1 = d.iloc[:,:9]
df2 = d.iloc[:,[0]+list(range(9,18))]
pd.plotting.scatter_matrix(df1,figsize=(13,13))
plt.show()
pd.plotting.scatter_matrix(df2,figsize=(13,13))
plt.show()


# ## Process outliers

# In[5]:


#sqft_livingとbedroomsに外れ値がありそうなので詳細確認
fig, ax = plt.subplots()
ax.scatter(x = d['sqft_living'], y = d['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel('sqft_living', fontsize=13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = d['bedrooms'], y = d['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel('bedrooms', fontsize=13)
plt.show()


# どちらも右下に価格の割に大きすぎるsqft_lotが見られる。外れ値と思われるため除外

# In[6]:


#Deleting outliers
d = d.drop(d[(d['sqft_living']>12000) & (d['price']<3000000)].index)
d = d.drop(d[(d['bedrooms']>30) & (d['price']<2000000)].index)


# ##目的変数の分布確認

# In[7]:


sns.distplot(d['price'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(d['price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(d['price'], plot=plt)
plt.show()


# 価格は右に型が長い分布で正規分布ではなさそう。線形回帰に対応するため正規分布に変換

# In[8]:


#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
d["price"] = np.log1p(d["price"])

#Check the new distribution 
sns.distplot(d['price'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(d['price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(d['price'], plot=plt)
plt.show()


# ## Feature Engineering

# In[9]:


# Confirm missing values
#print(d.isnull().any())
pd.DataFrame(d.isnull().sum(), columns=["num of missing"])


# It looks like the data has no missing values which means you do not have to worry about it:)

# ### Variables which will well explain price values.
# price shoud be highly correlated with sqft_living, grade, sqft_above, sqft_living15
# 
# That said, following parameters seem to have multicollinearity so let's check VIF.<br>
# bathrooms + sqft_living<br>
# grade + sqft_above + sqft_living15

# In[10]:


#Correlation map to see how features are correlated with SalePrice
corrmat = d.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# In[11]:


#VIFの計算
for cname in d.columns:  
    y=d[cname]
    X=d.drop(cname, axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    rsquared = regr.score(X,y)
    print(cname,":" ,1/(1-np.power(rsquared,2)))


# sqft_living and sqft_above has multicollinearity so I would remove sqft_above from the dataset.

# In[12]:


d = d.drop(['sqft_above'], axis=1)
for cname in d.columns:  
    y=d[cname]
    X=d.drop(cname, axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    rsquared = regr.score(X,y)
    print(cname,":" ,1/(1-np.power(rsquared,2)))


# The result shows no multicollinearity finally so let's try the simple linear regression result.

# In[13]:


X = d.drop(['price'],axis=1)
y = d['price']
regr = LinearRegression(fit_intercept=True).fit(X,y)
regr = LinearRegression(fit_intercept=True)
regr.fit(X, y)
print("決定係数=%s"%regr.score(X,y))
print("傾き=%s"%regr.coef_,"切片=%s"%regr.intercept_)
pd.Series(regr.coef_,index=X.columns).sort_values()  .plot(kind='barh',figsize=(6,8))


# As mentioned above, waterfront is highly correlated with the price and surprisingly lat shows strong impact. gradee, long, view, yr_renovated_bin, bathroooms and conditions are also important things to determine the price. 

# ### AIC comparison between each models

# In[14]:


def step_aic(model, exog, endog, **kwargs):
    """
    This select the best exogenous variables with AIC Both exog and endog values can be either str or list. (Endog list is for the Binomial family.)
    """

    # Convert　exog, endog　into list
    exog = np.r_[[exog]].flatten()
    endog = np.r_[[endog]].flatten()
    remaining = set(exog)
    selected = []  # The parameters which choosed the model

    #　AIC for only constant term
    formula_head = ' + '.join(endog) + ' ~ '
    formula = formula_head + '1'
    aic = model(formula=formula, **kwargs).fit().aic
    print('AIC: {}, formula: {}'.format(round(aic, 3), formula))
    current_score, best_new_score = np.ones(2) * aic

    # If AIC is not changed any more terminate the process
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:

            # Calculate the AIC, appending the each parameters
            formula_tail = ' + '.join(selected + [candidate])
            formula = formula_head + formula_tail
            aic = model(formula=formula, **kwargs).fit().aic
            print('AIC: {}, formula: {}'.format(round(aic, 3), formula))

            scores_with_candidates.append((aic, candidate))

        # Define the best candidate which has the lowest AIC
        scores_with_candidates.sort()
        scores_with_candidates.reverse()
        best_new_score, best_candidate = scores_with_candidates.pop()

        # If the new AIC score is lower than the potential best score, update the best score with the new parameter
        if best_new_score < current_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score

    formula = formula_head + ' + '.join(selected)
    print('The best formula: {}'.format(formula))
    return model(formula, **kwargs).fit()


# In[15]:


'''import statsmodels.formula.api as smf
model = step_aic(smf.ols, ['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_basement', 'yr_built', 'yr_renovated_bin', 
       'lat', 'long', 'sqft_living15','sqft_lot15'], ['price'],
      data=d)


# The best formula: <br>
# price ~ sqft_living + lat + view + grade + yr_built + waterfront + bedrooms + bathrooms + condition + sqft_basement + long + sqft_living15 + yr_renovated_bin + sqft_lot15 + sqft_lot <br>
# which means floor is not an important parameter.

# In[ ]:


X = d.drop(['floors'], axis=1)


# In[ ]:


#d['zipcode'] = d['zipcode'].apply(str)
d['bathrooms'] = d['bathrooms'].apply(int)
d['floors'] = d['floors'].apply(int)


# 分布の歪みを補正

# 対象変数に Box Cox 変換実施

# In[ ]:


# Check the skew of all numerical features
from scipy.stats import norm, skew 
numeric_feats = d.dtypes[d.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = d[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# ## cross validation

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
#まずは普通に推定

X_train,X_test,y_train,y_test = train_test_split(np.array(X),np.array(y),test_size=0.2,random_state=42)

# 必要なライブラリのインポート
from sklearn.ensemble import RandomForestRegressor
# モデル構築、パラメータはデフォルト
forest = RandomForestRegressor()
forest.fit(X_train, y_train)


# In[ ]:


# 予測値を計算
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

from sklearn.metrics import mean_squared_error
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)) )

from sklearn.metrics import r2_score
print('MSE train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) )
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)) )
print('MSE train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) )
#mae = mean_absolute_error(y_test, y_pred)
#print("MAE=%s"%round(mae,3) )


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
# パラメータ変更

forest = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=25,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=5,
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=4,
           oob_score=False, random_state=0, verbose=0, warm_start=False)
forest.fit(X_train, y_train)


# In[ ]:


# パラメータチューニング後の予測値を計算
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

from sklearn.metrics import mean_squared_error
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)) )

from sklearn.metrics import r2_score
print('MSE train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) )
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)) )
print('MSE train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) )


# testの精度が0.01向上

# # 他モデルとの比較

# 

# ## Define a cross validation strategy
# We use the cross_val_score function of Sklearn. However this function has not a shuffle attribut, we add then one line of code, in order to shuffle the dataset prior to cross-validation

# In[ ]:


#Validation function
n_folds = 2
#n_folds = 5


def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# ## Base models

# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# In[ ]:


#This model may be very sensitive to outliers. So we need to made it more robust on them. For that we use the sklearn's Robustscaler() method on pipeline
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
#Elastic Net Regression :again made robust to outliers
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
#Kernel Ridge Regression:
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
#Gradient Boosting Regression: Gradient Boosting Regression :With huber loss that makes it robust to outliers
GBoost = GradientBoostingRegressor()
#XGBoost
model_xgb = xgb.XGBRegressor()
#LightGBM
model_lgb = lgb.LGBMRegressor()


# ## Base models scores
# Let's see how these base models perform on the data by evaluating the cross-validation rmsle error

# In[ ]:


'''
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
'''


# ## Stacking models

# **Averaged base models class**

# In[ ]:


'''
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   
        


# We just average four models here **ENet, GBoost,  KRR and lasso**.  Of course we could easily add more models in the mix. 

# In[ ]:


'''
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# ###  Meta-modelの追加
# 上記の平均ベースモデルの予測にメタモデルを追加してトレーニングする。手順は以下の通り
# 1. トレーニングセットを2つの別々のセットに分割（例：** train **と** holdout **）
# 2. 最初の部分にいくつかの基本モデルを訓練 (**train**)
# 3. これらのベースモデルを2番目の部分(**holdout**)でテスト
# 4. 入力として3）の予測（out-of-folds予測と呼ぶ）を使用し、** meta-model **と呼ばれるより高いレベルのtrainerを訓練するための出力として正しい目的変数を使用
# 最初の3つのステップは繰り返し実行されます。例えば、5回スタッキングする場合、最初にトレーニングデータを5倍に分割します。次に、5回繰り返します。各反復では、すべての基本モデルを4回折りたたみ、残りのを予測
# 
# したがって、5回の反復の後に、データ全体をフォールドアウトの予測に使用し、ステップ4でメタモデルをトレーニングするための新しいフィーチャとして使用することが確実になる。予測部分については、テストデータ上のすべてのベースモデルの予測を平均し、それらを**meta feature**として使用して、最終的な予測をメタモデルで行う。

# In[ ]:


'''
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


# In[ ]:


stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)
'''
#重くてコンパイル不可
score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
'''


# ##  XGBoost and LightGBMを追加した累積回帰

# Add **XGBoost and LightGBM** to the** StackedRegressor** defined previously. 

# In[ ]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
stacked_averaged_models.fit(X_train, y_train)
stacked_train_pred = stacked_averaged_models.predict(X_train)
stacked_pred = np.expm1(stacked_averaged_models.predict(X_test))
print(rmsle(y_train, stacked_train_pred))


# ### Final Training and Prediction

# In[ ]:


'''#XGBoost
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))


# In[ ]:


'''#LightGBM
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))
'''


# In[ ]:


'''RMSE on the entire Train data when averaging

print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
'''


# **Ensemble prediction:**

# In[ ]:


#重くて実行不可
#ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15


# In[ ]:


'''def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
    '''


# In[ ]:


'''stacked_averaged_models.fit(d.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(X_train)
stacked_pred = np.expm1(stacked_averaged_models.predict(X_test))
print(rmsle(y_train, stacked_train_pred))
'''


# In[ ]:




