#!/usr/bin/env python
# coding: utf-8

# The  decision which model to use or what hyperparameters are most suitable is often based on some Cross-Validation technique producing an estimate of the out-of-sample prediction error $\bar{Err}$.
# 
# An alternative technique to produce an estimate of the out-of-sample error is the bootstrap, specifically the .632 estimator and the .632+ estimator mentioned in Elements of Statistical Learning. Surprisingly though, I could not find an implementation in sklearn.
# 
# Both techniques at first estimate an upward biased estimate of the prediction error $\hat{Err}$ and then reduce that bias differently. <br />
# $\hat{Err}$ is obtained through
# 
# $$\hat{Err} = \frac {1}{N} \displaystyle\sum_{i=1}^{N} \frac {1}{|C^{-i}|} \displaystyle\sum_{b \in {C^{-i}}} L(y_{i}, \hat{f}^{*b}(x_{i})).$$
# 
# Where
# * $N$ denotes the sample size.
# * $b$ denotes a specific bootstrap sample, whereas $B$ denotes the set of bootstrap samples.
# * $C^{-i}$ denotes the number of bootstrap samples $b$ where observation $i$ is not contained in.
# * $\hat{f}^{*b}(x_{i})$ denotes the estimated value of target $y_{i}$ by model $\hat{f}$ based on bootstrap sample $b$ and data $x_{i}$.
# * $L(y_{i}, \hat{f}^{*b}(x_{i}))$ denotes the loss-function between real value $y_{i}$ and estimated value $\hat{f}^{*b}(x_{i})$.
# 
# The pseudo-algorithm looks like this:
# 1. Create $B$ bootstrap samples $b$ with the same size $N$ as the original data <br />
# 2. For $i = 1, ..., N$ <br />
# I) &nbsp;&nbsp;&nbsp; For $b = 1, ..., B$ <br />
# Ia) &nbsp;&nbsp;&nbsp;&nbsp;If $i$ not in $b$ <br />
# Iai) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Estimate $\hat{f}^{*b}(x_{i})$ <br />
# Iaii) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute $L(y_{i}, \hat{f}^{*b}(x_{i}))$ <br />
# Ib) &nbsp;&nbsp;&nbsp;&nbsp;else next $b$ <br />
# II) &nbsp;&nbsp;Compute $\frac {1}{|C^{-i}|} \displaystyle\sum_{b \in {C^{-i}}} L(y_{i}, \hat{f}^{*b}(x_{i}))$ <br />
# 3. Compute $\frac {1}{N} \displaystyle\sum_{i=1}^{N} \frac {1}{|C^{-i}|} \displaystyle\sum_{b \in {C^{-i}}} L(y_{i}, \hat{f}^{*b}(x_{i}))$ 
# 
# The .632 estimator then calculates
# $$\bar{Err} = 0.632*\hat{Err} + 0.368*inSampleError$$,
# whereas the .632+ estimator demands a slightly more complex procedure to estimate $\bar{Err}$.
# However, due to its simplicity only the .632 estimator is presented in this kernel.
# 
# This is computationally intensive but when forced to work with a small data set where cross-validation is unreasonable. Estimating the test error through the bootstrap is  a viable option. 
# 
# After some brief data exploration and manipulation the above algorithm is implemented. Afterwards, the 5-fold cross-validation estimate of the test error is also computed and both are compared to the true test error.
# 
# In this kernel $\hat{f}$ is always represented by the linear regression and $L(y, \hat{f}(x))$ is represented by the MSE. 
# A reduced data set is used because the implementation in python is not very fast.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera

data = pd.read_csv('../input/kc_house_data.csv')

data = data.iloc[0:1000,:]

data.drop_duplicates('id', inplace=True)

print('Take a look at the data: \n', data.head(), '\n')

print('Examine data types of each predictor: \n', data.info(), '\n')

print('Check out summary statistics: \n', data.describe(), '\n')

print('Missing values?', data.columns.isnull().any(), '\n')

print('Columns names:', data.columns.values.tolist())


# In[ ]:


data = data.drop('zipcode', axis=1)
data = data.drop('date', axis=1)

nums = ['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement',
        'yr_built', 'sqft_living15', 'sqft_lot15']

numsData = data[nums]

numsData.hist(bins=50, figsize=(20,15))
plt.show()


# price, sqft_above, sqft_living, sqft_living15, sqft_lot, sqft_lot15 seem to be right-skewed and are transformed.
# In this case inverse-hyperbolic tranform is used, because, unlike log, it can handle zeros.
# Normally, one would re-transform the produced predictions of the target and the target itself before the loss-function is applied, however, in this case the scale of the target is not of interest.

# In[ ]:


def arcsinh(data, colList):
    for item in colList:
        data.loc[:,item] = np.arcsinh(data.loc[:,item].values)
    return data

jbCols = ['price', 'sqft_above', 'sqft_living', 'sqft_living15', 'sqft_lot', 'sqft_lot15']

numsData = arcsinh(numsData, jbCols)

numsData.hist(bins=50, figsize=(20,15))

data.loc[:,nums] = numsData


# Splitting data set and obtaining the $inSampleError$.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        data.drop('price', axis=1), data['price'], test_size=0.25, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(X_train, y_train)
inSamplePreds = lr.predict(X_train)
inSampleErr = mean_squared_error(inSamplePreds, y_train)


print('In-sample-error:', inSampleErr)


# Now, the Leave-One-Out Bootstrap function is implemented.
# It needs 4 arguments to be passed in. 
# 1. The data as a numpy array WITH an id-column, which uniquely identifies each observation, as the first column and 
# NO target column.
# 2. The target column as a numpy array.
# 3. The number of bootstrap samples to be created, and 
# 4. keyworded arguments of the model to be used.
# 
# While coding this function, it came to my mind that it is better to create $B$ bootstraped id-columns instead of $B$ complete data sets that all have to be stored in memory the whole time the function is running.
# This way, only the id-columns are stored all the time and each corresponding bootstrap data set is created through a JOIN-command as needed and then deleted when not in use anymore.
# However, because I could not get the numpy-JOIN to work as I wanted it to, the function unfortunately switches to pandas to execute the join command and then switches back to numpy.
# These cumbersome operations definitely do not improve the function's execution speed.

# In[ ]:


kwargs = {'fit_intercept': True, 'normalize': False, 'copy_X': True, 'n_jobs': 1}
# or kwargs = {}
def LOOB(data, targetCol, B_samples, **kwargs):
    avgLossVec = np.zeros((data.shape[0], 1))
    bootMat = np.zeros((data.shape[0], B_samples))
    idCol = np.zeros((data.shape[0], 1))
    idCol = data[:, 0]
    targetCol = np.stack((idCol, targetCol))
    targetCol = targetCol.T
    for column in range(bootMat.shape[1]):
        bootMat[:,column] = np.random.choice(idCol, idCol.shape[0],replace=True)
    for i in np.nditer(idCol):
        bootLossVec = np.zeros((1, 1))
        target = targetCol[targetCol[:,0]==i,1] 
        targetData = data[data[:,0]==i, 1:] 
        for column in range(bootMat.shape[1]):
            if i not in bootMat[:,column]:
                tempVec = pd.DataFrame(bootMat[:,column])
                tempVec.rename(columns={0:'id'}, inplace=True)
                tempData = pd.DataFrame(data)
                tempTarget = pd.DataFrame(targetCol)
                tempData.rename(columns={0:'id'}, inplace=True)
                tempTarget.rename(columns={0:'id'}, inplace=True)
                bootMat2 = tempVec.merge(tempData.drop_duplicates(subset=['id']), how='left', on='id')
                bootTarget = tempVec.merge(tempTarget.drop_duplicates(subset=['id']), how='left', on='id')
                del(tempVec)
                del(tempData)
                del(tempTarget)
                bootMat2 = bootMat2.iloc[:,1:].values
                bootTarget = bootTarget.iloc[:,1].values
                model = LinearRegression(kwargs)
                model.fit(bootMat2, bootTarget)
                prediction = model.predict(targetData)
                if column != 0:
                    bootLossVec = np.append(bootLossVec, mean_squared_error(target, prediction))
                elif column == 0:
                    bootLossVec[column] = mean_squared_error(target, prediction)
        avgLossVec[np.where(idCol == i)[0]] = np.mean(bootLossVec) 
    bootErr = np.mean(avgLossVec)
    return bootErr


bootErr = LOOB(X_train.values, y_train.values, 80, **kwargs)
bootError = bootErr*0.632 + inSampleErr*0.368

print('Bootstrap prediction error:', bootError)


# 5-Fold cross validation

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
mseee = make_scorer(mean_squared_error, greater_is_better=False)
cvScores = -cross_val_score(lr, X_train, y_train,cv=5 , scoring = mseee)
cvOutErr = cvScores.mean()

print('10-Fold error estimate:', cvOutErr)


# Out-of-Sample Error

# In[ ]:


testPreds = lr.predict(X_test)
trueError = mean_squared_error(testPreds, y_test)

print('True test error:', trueError)


# In[ ]:


bars = {'Bootstrap': bootError, '5-Fold-CV': cvOutErr, 'in Sample Error': inSampleErr, 
        'true test error': trueError}


fig = plt.figure()
plt.bar(range(len(bars)), bars.values(), align='center')
plt.xticks(range(len(bars)), bars.keys())

plt.show()

print(bars)


# As one can see above the bootstrap estimator is definitely an alternative, but an implementation in a quicker language would make it more applicable.
