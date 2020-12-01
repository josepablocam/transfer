
# coding: utf-8

# ## Lending Club - classification of loans
# 
# This project aims to analyze data for loans through 2007-2015 from Lending Club available on Kaggle. Dataset contains over 887 thousand observations and 74 variables among which one is describing the loan status. The goal is to create machine learning model to categorize the loans as good or bad. 
# 
# Contents:
# 
#     1. Preparing dataset for preprocessing
#     2. Reviewing variables - drop and edit
#     3. Missing values
#     4. Preparing dataset for modeling
#     5. Undersampling approach

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set(font_scale=1.6)

from sklearn.preprocessing import StandardScaler


# ### 1. Preparing dataset for preprocessing
# 
# In this part I will load data, briefly review the variables and prepare the 'y' value that will describe each loan as good or bad.

# In[2]:


data=pd.read_csv('../input/loan.csv',parse_dates=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)


# In[3]:


data.shape


# In[4]:


data.head()


# In[5]:


pd.value_counts(data.loan_status).to_frame().reset_index()


# There are 9 unique loan statuses. I will drop ones that are fully paid as these are historical entries. Next step will be to assign 0 (good) to Current loans and 1 (bad) to rest including: default and late loans, ones that were charged off or are in grace period.
# 
# First two are self-explanatory, charged off loan is a debt that is deemed unlikely to be collected by the creditor but the debt is not necessarily forgiven or written off entirely, a grace period is a provision in most loan contracts which allows payment to be received for a certain period of time after the actual due date.

# In[6]:


data = data[data.loan_status != 'Fully Paid']
data = data[data.loan_status != 'Does not meet the credit policy. Status:Fully Paid']


# In[7]:


data['rating'] = np.where((data.loan_status != 'Current'), 1, 0)


# In[8]:


pd.value_counts(data.rating).to_frame()


# In[9]:


print ('Bad Loan Ratio: %.2f%%'  % (data.rating.sum()/len(data)*100))


# The data is strongly imbalanced, however there are over 75 thousand bad loans that should suffice for a model to learn.

# In[10]:


data.info()


# ### 2. Reviewing variables - drop and edit
# 
# In this part I will review each non-numerical variable to either edit or drop it.

# There are two columns that describe a reason for the loan - title and purpose. As shown below title has many more categories which makes it less specific and helpful for the model, so it will be dropped.

# In[11]:


pd.value_counts(data.title).to_frame()


# In[12]:


pd.value_counts(data.purpose).to_frame()


# Application type variable shows whether the loan is individual or joint - number of joint loans will reflect huge number of NaN values in other variables dedicated for these loans.
# 
# Will change this variable to binary.

# In[13]:


pd.value_counts(data.application_type).to_frame()


# In[14]:


app_type={'INDIVIDUAL':0,'JOINT':1}
data.application_type.replace(app_type,inplace=True)


# In[15]:


pd.value_counts(data.term).to_frame()


# Term variable will be changed to numerical.

# In[16]:


term={' 36 months':36,' 60 months':60}
data.term.replace(term,inplace=True)


# Following two variables are dedicated to credit rating of each individual. Will change them to numerical while making sure that the hierarchy is taken into account. Lowest number will mean best grade/subgrade.

# In[17]:


pd.value_counts(data.grade).to_frame()


# In[18]:


grade=data.grade.unique()
grade.sort()
grade


# In[19]:


for x,e in enumerate(grade):
    data.grade.replace(to_replace=e,value=x,inplace=True)


# In[20]:


data.grade.unique()


# In[21]:


pd.value_counts(data.sub_grade).to_frame()


# In[22]:


sub_grade=data.sub_grade.unique()
sub_grade.sort()
sub_grade


# In[23]:


for x,e in enumerate(sub_grade):
    data.sub_grade.replace(to_replace=e,value=x,inplace=True)

data.sub_grade.unique()


# Following two variables describe title and length of employment. Title has 212 thousand categories so it will be dropped. Lenghth of employment should be sufficient to show whether an individual has a stable job.

# In[24]:


pd.value_counts(data.emp_title).to_frame()


# In[25]:


pd.value_counts(data.emp_length).to_frame()


# In[26]:


emp_len={'n/a':0,'< 1 year':1,'1 year':2,'2 years':3,'3 years':4,'4 years':5,'5 years':6,'6 years':7,'7 years':8,'8 years':9,'9 years':10,'10+ years':11}
data.emp_length.replace(emp_len,inplace=True)
data.emp_length=data.emp_length.replace(np.nan,0)
data.emp_length.unique()


# Home ownership variable should be informative for model as individuals who own their home should be much safer clients that ones that only rent it.

# In[27]:


pd.value_counts(data.home_ownership).to_frame()


# Verification status variable indicated whether the source of income of a client was verified.

# In[28]:


pd.value_counts(data.verification_status).to_frame()


# Payment plan variable will be dropped as it has only 3 'y' values.

# In[29]:


pd.value_counts(data.pymnt_plan).to_frame()


# Zip code information is to specific, there are 930 individual values, and there is no sense to make it more general as cutting it to two digits as this will only describe state, which does next veriable. Zip code will be dropped.

# In[30]:


pd.value_counts(data.zip_code).to_frame()


# In[31]:


pd.value_counts(data.addr_state).to_frame()


# Next variable is initial listing status of the loan. Possible values are – W, F and will be changed to binary.

# In[32]:


pd.value_counts(data.initial_list_status).to_frame()


# In[33]:


int_status={'w':0,'f':1}
data.initial_list_status.replace(int_status,inplace=True)


# Policy code has only 1 value so will be dropped.

# In[34]:


pd.value_counts(data.policy_code).to_frame()


# Recoveries variable informs about post charge off gross recovery. Will transform this to binary that will show whether this loan was recoveried. Will drop recovery fee as it is doubling similar information.

# In[35]:


pd.value_counts(data.recoveries).to_frame()


# In[36]:


data['recovery'] = np.where((data.recoveries != 0.00), 1, 0)


# In[37]:


pd.value_counts(data.collection_recovery_fee).to_frame()


# There are couple variables that can be transformed to date time.

# In[38]:


data.issue_d=pd.to_datetime(data.issue_d)


# In[39]:


earliest_cr_line=pd.to_datetime(data.earliest_cr_line)
data.earliest_cr_line=earliest_cr_line.dt.year


# In[40]:


data.last_pymnt_d=pd.to_datetime(data.last_pymnt_d)
data.next_pymnt_d=pd.to_datetime(data.next_pymnt_d)
data.last_credit_pull_d=pd.to_datetime(data.last_credit_pull_d)


# Dropping all variables mentioned above.

# In[41]:


data.drop(['id','member_id','desc','loan_status','url', 'title','collection_recovery_fee','recoveries','policy_code','zip_code','emp_title','pymnt_plan'],axis=1,inplace=True)


# In[42]:


data.head(10)


# ### 3. Missing values
# 
# There are observations that contain missing values, I will review and transform them variable by variable.

# Starting with defining a function to create a data frame of metadata containing count of null values and type.

# In[43]:


def meta (dataframe):
    metadata = []
    for f in data.columns:
    
        # Counting null values
        null = data[f].isnull().sum()
    
        # Defining the data type 
        dtype = data[f].dtype
    
        # Creating a Dict that contains all the metadata for the variable
        f_dict = {
            'varname': f,
            'nulls':null,
            'dtype': dtype
        }
        metadata.append(f_dict)

    meta = pd.DataFrame(metadata, columns=['varname','nulls', 'dtype'])
    meta.set_index('varname', inplace=True)
    meta=meta.sort_values(by=['nulls'],ascending=False)
    return meta


# In[44]:


meta(data)


# Variables: dti_joint, annual_inc_joint and verification_status_joint have so many null values as there are only 510 joint loans. Will replace NaN with 0 and 'None' for status.

# In[45]:


data.dti_joint=data.dti_joint.replace(np.nan,0)
data.annual_inc_joint=data.annual_inc_joint.replace(np.nan,0)
data.verification_status_joint=data.verification_status_joint.replace(np.nan,'None')


# Investigating variables connected to open_acc_6m which shows number of open trades in last 6 months. Variables open_il_6m, open_il_12m, open_il_24m, mths_since_rcnt_il, total_bal_il, il_util, open_rv_12m, open_rv_24m, max_bal_bc, all_util, inq_fi, total_cu_tl, inq_last_12m, collections_12_mths_ex_med have null values for the same rows - I will change them all to 0 as missing vaules show lack of open trades. 

# In[46]:


data.loc[(data.open_acc_6m.isnull())].info()


# In[47]:


variables1=['open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m','collections_12_mths_ex_med']

for e in variables1:
    data[e]=data[e].replace(np.nan,0)
    
meta(data)


# Variables containing month since last occurence of specific action have plenty null values that I understand as lack of the occurence.

# In[48]:


pd.value_counts(data.mths_since_last_record).unique()


# In[49]:


pd.value_counts(data.mths_since_last_major_derog).unique()


# In[50]:


pd.value_counts(data.mths_since_last_delinq).unique()


# Null values in these columns can't be replaced with 0 as it would mean that the last occurence was very recent. My understanding of these variables is that the key information is whether the specific action took place (delinquency, public record, worse rating), so I will turn these into binary categories of Yes (1), No (0).

# In[51]:


data.loc[(data.mths_since_last_delinq.notnull()),'delinq']=1
data.loc[(data.mths_since_last_delinq.isnull()),'delinq']=0

data.loc[(data.mths_since_last_major_derog.notnull()),'derog']=1
data.loc[(data.mths_since_last_major_derog.isnull()),'derog']=0

data.loc[(data.mths_since_last_record.notnull()),'public_record']=1
data.loc[(data.mths_since_last_record.isnull()),'public_record']=0

data.drop(['mths_since_last_delinq','mths_since_last_major_derog','mths_since_last_record'],axis=1,inplace=True)

meta(data)


# Investigating tot_coll_amt, tot_cur_bal, total_rev_hi_lim - these are three totals that have missing values for the same observations. I will change them to 0 as they should mean that the total is 0.

# In[52]:


data.loc[(data.tot_coll_amt.isnull())].info()


# In[53]:


variables2=['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']

for e in variables2:
    data[e]=data[e].replace(np.nan,0)
    
meta(data)


# Variable revol_util is revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.

# In[54]:


data.loc[(data.revol_util.isnull())].head(10)


# In[55]:


pd.value_counts(data.revol_util).to_frame()


# There is no clear answer to how to approach this variable, I will use 0 as this is the most common value and the amount of missing values is marginal.

# In[56]:


data.revol_util=data.revol_util.replace(np.nan,0)
    
meta(data)


# There are four datetime variables and three of them have missing values left. 
# 
# Variables last_credit_pull_d is the most recent month LC pulled credit for this loan, issue_d is the date loan was issued and next_payment_d is the date of next payment. There are not insightful variables so will be dropped.
# 
# I will check last_pymnt_d in more detail as this might have some predicitve value.

# In[57]:


pd.value_counts(data.last_pymnt_d).to_frame()


# In[58]:


late=data.loc[(data.last_pymnt_d=='2015-08-01')|(data.last_pymnt_d=='2015-09-01')|(data.last_pymnt_d=='2015-05-01')|(data.last_pymnt_d=='2015-06-01')]
pd.value_counts(late.rating).to_frame()


# This is clear information leak - model wouldn't have to learn, just check if last payment is late. I will transform this variable to binary category showing if any payment was received.

# In[59]:


data.loc[(data.last_pymnt_d.notnull()),'pymnt_received']=1
data.loc[(data.last_pymnt_d.isnull()),'pymnt_received']=0


# In[60]:


data.drop(['last_pymnt_d','issue_d','last_credit_pull_d','next_pymnt_d'],axis=1,inplace=True)

meta(data)


# There are seven variables with 3 missing values, this is such a small number that I will just replace NaN with most common values.

# In[61]:


variables3=['acc_now_delinq', 'open_acc', 'total_acc','pub_rec','delinq_2yrs','inq_last_6mths','earliest_cr_line']

for e in variables3:
    data[e]=data[e].replace(np.nan,data[e].mode()[0])
    
meta(data)


# There are no more missing values, so I can proceed to setting up machine learning model.

# ### 4. Preparing dataset for modeling
# 
# Standarization and transformation of non-numerical values.

# In[62]:


data.head()


# In[63]:


data.describe()


# Data needs to be standardized before applying any model as the numerical values have different ranges for different variables. 

# In[64]:


X=data.drop(['rating'],axis=1,inplace=False)
y=data.rating


# In[65]:


num_cols = X.columns[X.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
num_cols


# In[66]:


scaler=StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X.head()


# Next step is to use get dummies function that will transform all non numerical values to model-friendly format.

# In[67]:


X=pd.get_dummies(X,drop_first=True)


# In[68]:


X.head()


# In[69]:


X.shape


# ### 5. Undersampling approach
# 
# As the dataset is imbalanced (11% are bad loans), I want to test the approach of a repeated undersampling where each time model work on evenly distributed data.
# 
# I will use two functions, first one creates a confusion matrix and second one provides the repeated undersampling solution for tested models and prints out accuracy, recall and most importantly ROC AUC score.

# In[70]:


from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report,accuracy_score 


# In[71]:


import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[72]:


from copy import deepcopy

def cross_validate_repeated_undersampling_full(X, Y, model, n_estimators=3, cv=StratifiedKFold(5,random_state=1)):
    
    preds = []
    true_labels = []
        
    for train_index, test_index in cv.split(X,Y):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    
        scores = np.zeros((n_estimators,len(y_test)))
        for i in range(n_estimators):
            num1 = len(y_train[y_train==1])
            ind0 = np.random.choice(y_train.index[y_train==0], num1) 
            ind1 = y_train.index[y_train==1] 
            ind_final = np.r_[ind0, ind1]
            X_train_subsample = X_train.loc[ind_final]
            y_train_subsample = y_train.loc[ind_final]

            clf = deepcopy(model)
            clf.fit(X_train_subsample,y_train_subsample)  
            
            probs = clf.predict_proba(X_test)[:,1]
            scores[i,:] = probs

        preds_final = scores.mean(0) 
        preds.extend(preds_final)
        preds_labels=[round(x) for x in preds]
        
        true_labels.extend(y_test)
        
    cnf_matrix = confusion_matrix(true_labels,preds_labels)
    np.set_printoptions(precision=2)

    print("Accuracy score in the testing dataset: ", accuracy_score(true_labels,preds_labels))
    print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
        
    class_names = [0,1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix
                    , classes=class_names
                    , title='Confusion matrix')
    plt.show()
        
    print("ROC AUC score in the testing dataset: ", roc_auc_score(true_labels,preds))
        
    fpr, tpr, _ = roc_curve(true_labels,preds)
    roc_auc = auc(fpr, tpr)
        
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return


# In[ ]:


models=[['LogisticRegression',LogisticRegression()],['RandomForest',RandomForestClassifier()],['NaiveBayes',GaussianNB()],['LDA',LinearDiscriminantAnalysis()],['QDA',QuadraticDiscriminantAnalysis()]]


# In[ ]:


for e in models:
    print ("Testing:", e[0])
    cross_validate_repeated_undersampling_full(X, y, e[1])


# The best model is Logistic Regression with ROC AUC 0.862 followed by QDA with 0.85 and LDA with 0.84. Logistic Regression seems to perform better with all thresholds.
