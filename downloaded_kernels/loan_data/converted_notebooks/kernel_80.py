
# coding: utf-8

# **In this simple kernel, I will attempt to predict whether customers will be "Charged Off" on a loan using Random Forests Classifier.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


loan  = pd.read_csv("../input/loan.csv")

print(loan.head())


# First let's see how much Nans are there per column

# In[ ]:


#determine nan percentage
check_null = loan.isnull().sum().sort_values(ascending=False)/len(loan)

#print all with 20% NaNs
print(check_null[check_null > 0.2])


# In[ ]:


#loads of columns ... so let's remove these
loan.drop(check_null[check_null > 0.2].index, axis=1, inplace=True)
loan.dropna(axis=0, thresh=30,inplace=True)


# After culling the NaN dominated columns, there are still a lot of features. Some will have useful info, others not. At this point I carefully weeded out any column that I think may be well useless. My main criteria is whether a feature is dominated by a single value (> 80%)
# 1. id and member_id: somehow I don't think these will be useful, condidering all were unique
# 2. Policy_cose: this is the same for all customers
# 3. url: this is the webpage of the loan data. May come in handy at someother stage (maybe)
# 4. zip_code and addr_state: I really don't think that the state and location of aperson will determine if they will repay a loan. Although, I could be wrong ....
# 5. application_type: was >99% INDIVIDUAL
# 6. 'pymnt_plan': 99.99% N
# 7. emp_title: this could be useful. Possbly through NLP. 
# 8. acc_now_delinq: > 99% 0
# 9. title: may be very useful. Requires NLP
# 10. collections_12_mths_ex_med: ~98% 0
# 11. collection_recovery_fee > 98% 0 

# In[ ]:


#first let's remove some columns
del_cols = ['id', 'member_id', 'policy_code', 'url', 'zip_code', 'addr_state',
            'pymnt_plan','emp_title','application_type','acc_now_delinq','title',
            'collections_12_mths_ex_med','collection_recovery_fee']


# In[ ]:


loan = loan.drop(del_cols,axis=1)


# The point of this exercise is to predict if a loan will be "Charged Off". Let's see the breakdown of the target column: 'loan_status'

# In[ ]:


print(loan['loan_status'].value_counts()/len(loan))


# Yikes! Ok for now we will ignore "Current" customers. Note, we could use the model generated to predict whether a "Current" customers will be "Charged Off". 

# In[ ]:


loan = loan[loan['loan_status'] != 'Current']


# In[ ]:


print(loan['loan_status'].value_counts()/len(loan))


# The column 'emp_length' may be useful

# In[ ]:


print(loan['emp_length'].unique())


# Let's convert this to categorical data

# In[ ]:


loan['empl_exp'] = 'experienced'
loan.loc[loan['emp_length'] == '< 1 year', 'empl_exp'] = 'inexp'

loan.loc[loan['emp_length'] == '1 year', 'empl_exp'] = 'new'
loan.loc[loan['emp_length'] == '2 years', 'empl_exp'] = 'new'            
loan.loc[loan['emp_length'] == '3 years', 'empl_exp'] = 'new'

loan.loc[loan['emp_length'] == '4 years', 'empl_exp'] = 'intermed'
loan.loc[loan['emp_length'] == '5 years', 'empl_exp'] = 'intermed'
loan.loc[loan['emp_length'] == '6 years', 'empl_exp'] = 'intermed'

loan.loc[loan['emp_length'] == '7 years', 'empl_exp'] = 'seasoned'
loan.loc[loan['emp_length'] == '8 years', 'empl_exp'] = 'seasoned'
loan.loc[loan['emp_length'] == '9 years', 'empl_exp'] = 'seasoned'

loan.loc[loan['emp_length'] == 'n/a', 'empl_exp'] = 'unknown'

#delete the emp_length column 
loan = loan.drop('emp_length',axis=1)


# In[ ]:


#remove all rows with nans
loan.dropna(axis=0, how = 'any', inplace = True)


# In[ ]:


print(loan['loan_status'].value_counts()/len(loan))


# In[ ]:


#extract the target column and convert to Charged Off to 1 and the rest as 0
mask = (loan.loan_status == 'Charged Off')
loan['target'] = 0
loan.loc[mask,'target'] = 1

target = loan['target']
loan = loan.drop(['loan_status','target'],axis=1)


# In[ ]:


target.value_counts()


# The next step is to convert all categorical data to dummy numerical data. First let's seperate the categorical from number columns

# In[ ]:


loan_categorical = loan.select_dtypes(include=['object'], exclude=['float64','int64'])
features = loan.select_dtypes(include=['float64','int64'])


# In[ ]:


#one-hot-encode the categorical variables and combine with the numercal values
for col in list(loan_categorical):
    dummy = pd.get_dummies(loan_categorical[col])
    features = pd.concat([features,dummy],axis=1)


# In[ ]:


#time to split and build models
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features,target)


# The model we will build is Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


RF = RandomForestClassifier(n_estimators=500)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
print('Test score: {:.2f}'.format(RF.score(X_test, y_test)))
print("Confusion matrix:\n%s" % confusion_matrix(y_test, y_pred))
print("Classification report for Random Forest classifier %s:\n%s\n"
      % (RF, classification_report(y_test, y_pred)))


# Nice! Carefully selecting features as well as some feature engineering paid off! 100% precision and 98% Recall for all "Charged off" loans! Since the dataset is skewed, let's have a llok at the Precision and Recall curve

# In[ ]:


precision, recall, thresholds = precision_recall_curve(y_test,RF.predict_proba(X_test)[:, 1])
AUC = average_precision_score(y_test, RF.predict_proba(X_test)[:, 1])

plt.plot(precision, recall, label='AUC: {:.2f} for {} Trees'.format(AUC, 500))
close_default_rf = np.argmin(np.abs(thresholds - 0.5))
plt.plot(precision[close_default_rf], recall[close_default_rf], 'o', c='k',
         markersize=10, fillstyle="none", mew=2)

plt.ylabel('Recall')
plt.xlabel('Precision')
plt.title('Precision-Recall Curve Random Forest')
plt.legend(loc='best')
plt.show()


# Next: The next step? We can use this model to determine the probability any of the "Current" customers will be "Charged Off". 
