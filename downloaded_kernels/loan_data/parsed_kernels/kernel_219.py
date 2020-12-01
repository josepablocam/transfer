
# coding: utf-8

# # Analyzing LendingClub data
# 
# 
# Solve the problem of classifying loans for refundable and non-refundable.
# 
# Target feature: loan_status (current loan status).
# 
# Quality metrics: Accuracy and Area under the ROC curve.

# In[3]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from sklearn.preprocessing import Imputer
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import operator
from xgboost import XGBClassifier


# In[4]:


df = pd.read_csv('../input/loan.csv')


# In[5]:


df.shape


# In[6]:


df.head()


# # Data preprocessing

# Let's look at loan status. 

# In[7]:


df.loan_status.value_counts()


# We will take only FullyPaid and Charged Off statuses.

# In[8]:


df = df[(df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off')]


# The fields 'id' and 'member_id' contain only one unique value, so it is not interesting to us. The 'url' field does not add any knowledge to solve the problem. 
# Many fields contain only one value.
# 

# In[10]:


print(np.count_nonzero(df['id'].unique()))
print(np.count_nonzero(df['member_id'].unique()))


# In[9]:


print(np.count_nonzero(df['pymnt_plan'].unique()))


# In[10]:


df = df[df['pymnt_plan'] == 'n']


# In[11]:


print(np.count_nonzero(df['application_type'].unique()))


# In[12]:


df = df[df['application_type'] == 'INDIVIDUAL']


# In[13]:


df.iloc[:, 20:30]


# The fields 'out_prncp', 'next_pymnt_d' and 'out_prncp_inv' are greater than 0 only for cases that we do not consider. They can also be thrown out of data.

# In[14]:


df[['out_prncp','out_prncp_inv', 'loan_status']][(df['out_prncp']>0) | (df['out_prncp_inv']>0)]


# In[15]:


df[['next_pymnt_d', 'loan_status']][df.next_pymnt_d.notnull()]


# In[16]:


df[['policy_code', 'annual_inc_joint', 'dti_joint']].describe()


# It's worth noting that the 'grade' column is redundant, because it is completely duplicated in the 'sub_grade' column.

# In[17]:


df1 = df.drop(columns=['policy_code', 'next_pymnt_d', 'out_prncp','out_prncp_inv', 'pymnt_plan', 
                       'initial_list_status', 'member_id', 'id', 'url', 'application_type',
                       'grade', 'annual_inc_joint', 'dti_joint'])


# In[18]:


df1.shape


# Let's select the digital value from the loan conditions ('term' field), interest rate on the loan ('int_rate' field) and employment in years ('emp_length').

# In[19]:


terms = []
for row in df1.term:
    terms.append(re.findall('\d+', row)[0])


# In[20]:


df1.term = terms


# In[21]:


emp_lengths = []
for row in df1.emp_length:
    if pd.isnull(row) == False:
        emp_lengths.append(re.findall('\d+', row)[0])
    else:
        emp_lengths.append(row)


# In[22]:


df1.emp_length = emp_lengths


# In[23]:


df1.iloc[:, 60:70]


# ** Processing of missing values: **
# 
# 
# We will fill them with an average, median or most common attribute, depending on the nature of the data.
# 
# We will create the features indicating the lack of data.

# Count the number of nan values by columns:

# In[24]:


for col in df1.columns:
    nan_count = np.sum(pd.isnull(df1[col]))
    if nan_count != 0:
        print(col, " : ", np.sum(pd.isnull(df1[col])))


# Delete the columns where missing values are the vast majority.

# In[25]:


df1 = df1.drop(columns=['verification_status_joint', 'open_acc_6m', 'open_il_6m',
                       'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 
                      'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 
                      'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m' ])


# In[26]:


df1.shape


# In[27]:


for col in df1.columns:
    nan_count = np.sum(pd.isnull(df1[col]))
    if nan_count != 0:
        print(col, " : ", np.sum(pd.isnull(df1[col])))


# In[28]:


df1 = df1.drop(columns = ['mths_since_last_major_derog'])


# In[29]:


df1.tot_coll_amt = df1.tot_coll_amt.replace(np.nan, 0)


# In[30]:


imp = Imputer(strategy='median')
df1.total_rev_hi_lim = imp.fit_transform(df1.total_rev_hi_lim.reshape(-1, 1))


# In[31]:


df1.tot_cur_bal = df1.tot_cur_bal.replace(np.nan, 0)


# In[32]:


imp = Imputer(strategy='most_frequent')
df1.collections_12_mths_ex_med = imp.fit_transform(df1.collections_12_mths_ex_med.reshape(-1, 1))


# In[33]:


df1['mths_since_last_delinq_nan'] =  np.isnan(df1.mths_since_last_delinq)*1


# In[34]:


imp = Imputer(strategy='most_frequent')
msld = imp.fit_transform(df1.mths_since_last_delinq.values.reshape(-1, 1))
df1.mths_since_last_delinq = msld


# In[35]:


df1.mths_since_last_record.hist()


# In[36]:


df1['mths_since_last_record_nan'] =  np.isnan(df1.mths_since_last_record)*1


# In[37]:


imp = Imputer(strategy='median')
mslr = imp.fit_transform(df1.mths_since_last_record.values.reshape(-1, 1))
df1.mths_since_last_record = mslr


# In[38]:


df1['revol_util_nan'] =  pd.isnull(df1.revol_util)*1


# In[39]:


imp = Imputer(strategy='mean')
df1.revol_util = imp.fit_transform(df1.revol_util.values.reshape(-1, 1))


# In[40]:


df1['emp_length_nan'] =  pd.isnull(df1.emp_length)*1


# In[41]:


imp = Imputer(strategy='median')
df1.emp_length = imp.fit_transform(df1.emp_length.values.reshape(-1, 1))


# In[42]:


for col in df1.columns:
    nan_count = np.sum(pd.isnull(df1[col]))
    if nan_count != 0:
        print(col, " : ", np.sum(pd.isnull(df1[col])))


# In[43]:


#все категориальные признаки
col_cat = '''
sub_grade
home_ownership
verification_status
purpose
zip_code
addr_state
'''.split()


# Apply LabelEncoder to categorical data.

# In[44]:


lbl_enc = LabelEncoder()
for x in col_cat:
    df1[x+'_old'] = df[x]
    df1[x] = lbl_enc.fit_transform(df1[x])


# We collect the text attributes in one column and apply TfIDf to them.

# In[45]:


df1['text'] = df1.emp_title + ' ' + df1.title + ' ' + df1.desc
df1['text'] = df1['text'].fillna('nan')


# In[46]:


tfidf = TfidfVectorizer()
df_text = tfidf.fit_transform(df1['text'])


# Process the columns with the date.

# In[47]:


df1.issue_d = pd.to_datetime(df1.issue_d, format='%b-%Y')


# In[48]:


df1['issue_d_year'] =df1.issue_d.dt.year


# In[49]:


df1['issue_d_month'] =df1.issue_d.dt.month


# In[50]:


df1.earliest_cr_line = pd.to_datetime(df1.earliest_cr_line, format='%b-%Y')


# In[51]:


df1['earliest_cr_line_year'] =df1.earliest_cr_line.dt.year
df1['earliest_cr_line_month'] =df1.earliest_cr_line.dt.month


# The information in the columns:
# funded_amnt,
# funded_amnt_inv,
# total_pymnt,
# total_pymnt_inv,
# total_rec_prncp,
# total_rec_int,
# last_pymnt_d,
# last_pymnt_amnt,
# last_credit_pull_d,
# recoveries,
# collection_recovery_fee
# not known at the time of the conclusion of the transaction, so it is required to remove the model from consideration.

# In[52]:


#real features
col_int_float2 = '''loan_amnt
term
int_rate
installment
emp_length
annual_inc
dti
delinq_2yrs
inq_last_6mths
mths_since_last_delinq
mths_since_last_record
open_acc
pub_rec
revol_bal
revol_util
total_acc
collections_12_mths_ex_med
acc_now_delinq
tot_coll_amt
tot_cur_bal
total_rev_hi_lim
mths_since_last_delinq_nan
mths_since_last_record_nan
revol_util_nan
emp_length_nan
issue_d_year
issue_d_month
earliest_cr_line_year
earliest_cr_line_month
'''.split()


# In[53]:


df1[col_int_float2 + col_cat].head()


# In[54]:


df1['term'] = df1.term.astype(str).astype(int)


# In[55]:


df1['int_rate'] = df1.int_rate.astype(str).astype(float)


# Let's see what the result will be provided by Logistic regression on this data.

# In[56]:


df2 = df1[(df1['loan_status'] == 'Fully Paid') | (df1['loan_status'] =='Charged Off') ]


# In[57]:


targets = []
for row in df2.loan_status:
    if row == 'Fully Paid':
        targets.append(1)
    else:
        targets.append(0)


# In[58]:


df2['target'] = targets


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(df2[col_int_float2 + col_cat], df2['target'], 
                                                    test_size=0.3, random_state=42, stratify=df2['target'])


# # Data imbalance
# 
# If you build a model on such a data partition, the classifier will only predict 'Fully Paid' cases. We will always receive high accuracy, since these cases in the control sample are about 82%.

# In[60]:


sum(targets)/len(targets)


# In[61]:


def classify(est, x, y):

    est.fit(x, y)

    y2 = est.predict_proba(X_test)
    y1 = est.predict(X_test)

    print("Accuracy: ", metrics.accuracy_score(y_test, y1))
    print("Area under the ROC curve: ", metrics.roc_auc_score(y_test, y2[:, 1]))

    print("F-metric: ", metrics.f1_score(y_test, y1))
    print(" ")
    print("Classification report:")
    print(metrics.classification_report(y_test, y1))
    print(" ")
    print("Evaluation by cross-validation:")
    print(cross_val_score(est, x, y))
    
    return est, y1, y2[:, 1]


# In[62]:


xgb0, y_pred_b, y_pred2_b = classify(XGBClassifier(), X_train, y_train)


# Negative examples for quality education are not enough. If you build a classifier on these data, it will produce 99% of all labels as Fully Paid.
# 
# At the same time, we see a high quality on a deferred sample: an accuracy of 85%. However, the completeness of the Charged off is 0.
# 
# In the predicted tags, almost all data refer to 'Fully Paid':

# In[63]:


sum(y_pred_b)/len(y_pred_b)


# It is necessary to break the training sample into several sets in which the data will be balanced. Train different classifiers on them and average the result.
# 
# At the same time data related to Charged off will be repeated in each set, because they are few.

# In[64]:


X_train[y_train==1].shape


# In[65]:


X_train[y_train==0].shape


# In[66]:


145404/31673


# In[67]:


64+32+32


# In[68]:


X_train2 = X_train.drop(X_train[y_train==1].iloc[32000:].index)
y_train2 = y_train[X_train2.index]

X_train3 = X_train.drop(X_train[y_train==1].iloc[0:32000].index)
X_train3 = X_train3.drop(X_train3[y_train==1].iloc[32000:].index)
y_train3 = y_train[X_train3.index]

X_train4 = X_train.drop(X_train[y_train==1].iloc[0:64000].index)
X_train4 = X_train4.drop(X_train4[y_train==1].iloc[32000:].index)
y_train4 = y_train[X_train4.index]

X_train5 = X_train.drop(X_train[y_train==1].iloc[0:96000].index)
X_train5 = X_train5.drop(X_train5[y_train==1].iloc[32000:].index)
y_train5 = y_train[X_train5.index]

X_train6 = X_train.drop(X_train[y_train==1].iloc[0:128000].index)
X_train6 = X_train6.drop(X_train6[y_train==1].iloc[32000:].index)
y_train6 = y_train[X_train6.index]


# In[69]:


xgb = XGBClassifier(n_estimators=47, learning_rate=0.015)


# In[70]:


xgb1, y_pred, y_pred2 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train2, y_train2)


# In[71]:


xgb2, y_pred_3, y_pred2_3 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train3, y_train3)


# In[72]:


xgb3, y_pred_4, y_pred2_4 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train4, y_train4)


# In[73]:


xgb4, y_pred_5, y_pred2_5 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train5, y_train5)


# In[74]:


xgb5, y_pred_6, y_pred2_6 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train6, y_train6)


# We average the results of the five classifiers.

# In[75]:


y_avg = (y_pred + y_pred_3 + y_pred_4 + y_pred_5 + y_pred_6)/5
y_avg = (y_avg>0.3)*1
print("Accuracy: ", metrics.accuracy_score(y_test, y_avg))
print("F-metric: ", metrics.f1_score(y_test, y_avg))

print(" ")
print("Classification report:")
print(metrics.classification_report(y_test, y_avg))


# # Let's try other classifiers

# In[76]:


from sklearn.linear_model import RidgeClassifier, PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


# In[77]:


knc, y_p, y_p2 = classify(KNeighborsClassifier(), X_train2, y_train2)


# In[78]:


logit, y_p, y_p2 = classify(LogisticRegression(), X_train2, y_train2)


# In[79]:


bnb, y_p, y_p2 = classify(BernoulliNB(), X_train2, y_train2)


# In[80]:


dtc, y_p, y_p2 = classify(DecisionTreeClassifier(), X_train2, y_train2)


# In[81]:


xg, y_p, y_p2 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train2, y_train2)


# # Analysis of the impact of features

# In[82]:


def feat_importance(estimator):
    feature_importance = {}
    for index, name in enumerate(df2[col_int_float2 + col_cat].columns):
        feature_importance[name] = estimator.feature_importances_[index]

    feature_importance = {k: v for k, v in feature_importance.items()}
    sorted_x = sorted(feature_importance.items(), key=operator.itemgetter(1), reverse = True)
    
    return sorted_x


# In[83]:


feat1 = feat_importance(xgb1)
feat1[:12]


# In[84]:


feat2 = feat_importance(xgb2)
feat2[:12]


# In[85]:


feat3 = feat_importance(xgb3)
feat3[:12]


# In[86]:


feat4 = feat_importance(xgb4)
feat4[:12]


# In[87]:


feat5 = feat_importance(xgb5)
feat5[:12]


# Features that are in all 5 classifiers: annual_inc, int_rate, term, dti, issue_d_year.
# Let's try to leave only them and look at the result.

# In[88]:


col_xgb = '''annual_inc
int_rate
term
dti
issue_d_year
'''.split()


# # We will try to use only selected features

# In[89]:


X_train, X_test, y_train, y_test = train_test_split(df2[col_xgb], df2['target'], 
                                                    test_size=0.3, random_state=42, stratify=df2['target'])
X_train2 = X_train.drop(X_train[y_train==1].iloc[32000:].index)
y_train2 = y_train[X_train2.index]

X_train3 = X_train.drop(X_train[y_train==1].iloc[0:32000].index)
X_train3 = X_train3.drop(X_train3[y_train==1].iloc[32000:].index)
y_train3 = y_train[X_train3.index]

X_train4 = X_train.drop(X_train[y_train==1].iloc[0:64000].index)
X_train4 = X_train4.drop(X_train4[y_train==1].iloc[32000:].index)
y_train4 = y_train[X_train4.index]

X_train5 = X_train.drop(X_train[y_train==1].iloc[0:96000].index)
X_train5 = X_train5.drop(X_train5[y_train==1].iloc[32000:].index)
y_train5 = y_train[X_train5.index]

X_train6 = X_train.drop(X_train[y_train==1].iloc[0:128000].index)
X_train6 = X_train6.drop(X_train6[y_train==1].iloc[32000:].index)
y_train6 = y_train[X_train6.index]


# In[90]:


xgb1, y_pred, y_pred2 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train2, y_train2)


# In[91]:


xgb2, y_pred_3, y_pred2_3 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train3, y_train3)


# In[92]:


xgb3, y_pred_4, y_pred2_4 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train4, y_train4)


# In[93]:


xgb4, y_pred_5, y_pred2_5 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train5, y_train5)


# In[94]:


xgb5, y_pred_6, y_pred2_6 = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train6, y_train6)


# We average the results of the five classifiers.

# In[95]:


y_avg = (y_pred + y_pred_3 + y_pred_4 + y_pred_5 + y_pred_6)/5
y_avg = (y_avg>0.3)*1
print("Accuracy: ", metrics.accuracy_score(y_test, y_avg))
print("F-metric: ", metrics.f1_score(y_test, y_avg))

print(" ")
print("Classification report:")
print(metrics.classification_report(y_test, y_avg))


# In[96]:


col_xgb = '''annual_inc
int_rate
term
dti
issue_d_year
'''.split()


# In[97]:


y_avg2 = (y_pred2 + y_pred2_3 + y_pred2_4 + y_pred2_5 + y_pred2_6)/5
print("Area under the ROC curve: ", metrics.roc_auc_score(y_test, y_avg2))


# As we see the quality of the classification has not changed, so in the model you can only leave these signs:
# 
# Annual income
# 
# Interest rate on the loan
# 
# A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower's self-reported monthly income.
# 
# Credit term
# 
# The year that the loan was funded

# Let's try to consider these features.
# 
# # The annual income of the borrower

# In[98]:


ax = df2[['annual_inc', 'target']][df2.annual_inc < 100000].boxplot(by='target', figsize=(6, 5), vert=False )
ax.set_yticklabels(['Charged Off', 'Fully Paid'])
ax.set_title('Annual income of the borrower')


# The annual income of those who do not return a loan is slightly less than those who return.

# # Interest rate of the loan

# The probability of a loan repayment with a lower rate is higher than a higher one.

# In[99]:


ax = df2[['int_rate']][df2.target==1].hist(bins=20, normed=True, alpha=0.8, figsize=(6,4), label = u'Fully Paid')
df2[['int_rate']][df2.target==0].hist(ax=ax, bins=20, normed=True, alpha=0.5, label = u'Charged off')
plt.title("Interest rate of the loan")
plt.xlabel('Interest rate')
plt.legend()


# # DTI

# In[100]:


ax = df2['dti'][df2.target==1].hist(bins=20, normed=True, alpha=0.8, figsize=(8,4), label = u'Fully Paid')
df2['dti'][df2.target==0].hist(ax=ax, bins=20, normed=True, alpha=0.5, label = u'Charged off')
plt.title("DTI")
plt.legend(loc='best', frameon=False)


# # Term of the loan (36 or 60 months)

# In[316]:


ax = df2['term'][df2.target==1].hist(bins=2, normed=True, alpha=0.8, figsize=(4,4), label = u'Fully Paid')
df2['term'][df2.target==0].hist(ax=ax, bins=2, normed=True, alpha=0.5, label = u'Charged off')
ax.set_xticklabels(['', '', '36 month', '', '', '60 month'])
plt.title("Term of the loan")
plt.legend(loc='best', frameon=False)
ax.tick_params(axis=x, pad=10)


# A loan taken for a longer period is less likely to be returned.

# # The year which the loan was funded

# In[101]:


ax = df2['issue_d_year'][df2.target==1].hist(bins=20, normed=True, alpha=0.8, figsize=(6,4),  label = u'Fully Paid')
df2['issue_d_year'][df2.target==0].hist(ax=ax, bins=20, normed=True, alpha=0.5, label = u'Charged off')
plt.title("The year which the loan was funded")
plt.legend(loc='best', frameon=False)


# # Conclusions about the applicability of the model
# 
# 
# The model does not provide a sufficient level of quality to use it in automatic mode.
# 
# The accuracy is 71.8%
# 
# 
# The area under the curve is 68.3%
# 
# 
# This model can be used in a semi-automatic mode, as a primary recommendation for an expert.
