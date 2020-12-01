
# coding: utf-8

# We are going to try and predict the if a loan will be late or default using the below data. The do the preprocessing and to explore the data.

# ### Import Libraries

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white', font_scale=0.9)


# ### Import DataSet

# In[ ]:


dataset = pd.read_csv('../input/loan.csv')


# In[ ]:


pd.set_option('display.max_columns', len(dataset.columns))
dataset.head(3)


# In[ ]:


pd.reset_option('display.max_columns')


# ### Fix missing values - Categorical

# For the categorical variables we are just going to replace NaN with 'Unknown'. 
# 
# We will fill verification_status_joint using the value in verification_status as these are all individual applications and these values are not filled out. 

# In[ ]:


dataset['verification_status_joint'].fillna(dataset['verification_status'], inplace=True)


# In[ ]:


strColumns = dataset.select_dtypes(include=['object']).columns.values
dataset[strColumns] = dataset[strColumns].fillna('Unknown')


# Check that all the NaN values have been replaced

# In[ ]:


dataset.select_dtypes(exclude=[np.number]).isnull().sum()


# ### Fix missing values - Numeric

# First we will check the number of missing values for each of the columns

# In[ ]:


dataset.select_dtypes(include=[np.number]).isnull().sum()


# The first columns that we are going to update are annual_inc_joint, dti_joint and verification_status_joint. For individual accounts these are blank but we want to use the joint values so we will populate these with the individual values for individual accounts.

# In[ ]:


dataset[dataset['application_type'] != 'INDIVIDUAL']['annual_inc_joint'].isnull().sum()


# In[ ]:


dataset['annual_inc_joint'].fillna(dataset['annual_inc'], inplace=True)
dataset['dti_joint'].fillna(dataset['dti'], inplace=True)


# For the remainder of the missing values we are going to fix the missing values by replacing any NaN values with the mean values

# In[ ]:


strColumns = dataset.select_dtypes(include=[np.number]).columns.values
dataset[strColumns] = dataset[strColumns].fillna(dataset[strColumns].mean())


# Again check that there are no more NaN values

# In[ ]:


dataset.select_dtypes(include=[np.number]).isnull().sum()


# ### Create variable for default

# The loan status field has a number of different values. We are going to group the defaulted into a single category. The current loans will also be removed from the dataset as we are unable to predict these one way or the other yet. 

# In[ ]:


dataset['loan_status'].value_counts()


# First we are going to remove the issued and does not meet credit policy loans as these are either brand new loans or loans that didn't meet the credit policy and were forced to be closed. We can't learn much from them in terms of predicting whether the client will default by themselves.

# In[ ]:


dataset = dataset[~dataset['loan_status'].isin(['Issued',
                                 'Does not meet the credit policy. Status:Fully Paid',
                                 'Does not meet the credit policy. Status:Charged Off'
                                ])]


# Next we are going to create a default. 
# 
# In grace period is technically a late payment but for this we are not going to include it as a default as these include timing of payments being a day late etc. We are more interested in predicting the loans that will be significantly late with a payment and eventually default on the loan. These are the loans and may have to be written off or sent to a collection agency.

# In[ ]:


def CreateDefault(Loan_Status):
    if Loan_Status in ['Current', 'Fully Paid', 'In Grace Period']:
        return 0
    else:
        return 1 
    
dataset['Default'] = dataset['loan_status'].apply(lambda x: CreateDefault(x))


# ### Exploring other categorical variables

# Next we are going to look at a few of the other categorical variables to see how many labels they have

# In[ ]:


dataset['term'].value_counts()


# The only two terms of loans that are provided by LC are either a 3 or 5 year loan.

# In[ ]:


dataset['emp_length'].value_counts()


# For employment length we are going to convert this into a number field. For n/a we are going to include these along with the < 1 year as we can't be sure of the length of employment. This is because if a new customer was to not enter anything then we have to assume that they are not employed when predicting actual loans going forward.

# In[ ]:


def EmpLength(emp_len):
    if emp_len[:2] == '10':
        return 10
    elif emp_len[:1] in ['<', 'n']:
        return 0
    else:
        return int(emp_len[:1])
    
dataset['Emp_Length_Years'] = dataset['emp_length'].apply(lambda x: EmpLength(x))


# In[ ]:


dataset['purpose'].value_counts()


# The vast majority of loans provided by LC are debt consolidation both of other loans and credit card debt.

# In[ ]:


dataset['grade'].value_counts().sort_index()


# The last column that we want to create is a year earliest credit line. We also need to update the records that were set to unknown to the mean. We will round the mean so that it is a full year

# In[ ]:


dataset['Earliest_Cr_Line_Yr'] = pd.to_numeric(dataset['earliest_cr_line'].str[-4:], errors='coerce').round(0)


# In[ ]:


dataset['Earliest_Cr_Line_Yr'].isnull().sum()


# In[ ]:


dataset['Earliest_Cr_Line_Yr'] = dataset['Earliest_Cr_Line_Yr'].fillna(int(dataset['Earliest_Cr_Line_Yr'].mean()))


# In[ ]:


dataset['Earliest_Cr_Line_Yr'].isnull().sum()


# ### Exploring the relationship of variables to late payment

# The first thing I'm keen to look at is the grade to see if the grade of the loan actually correlates to whether a loan will have a late payment or not.

# #### Grade

# In[ ]:


nNoLate = len(dataset[dataset['Default'] == 0])
nLate = len(dataset[dataset['Default'] == 1])

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(9, 3))

sns.barplot(x='grade', y='id', data=dataset, 
            estimator=lambda x: len(x) / (nLate + nNoLate) * 100,
            ax=ax1, order=sorted(dataset['grade'].unique()), palette='deep')
sns.barplot(x='grade', y='id', data=dataset[dataset['Default'] == 0], 
            estimator=lambda x: len(x) / nNoLate * 100,
            ax=ax2, order=sorted(dataset['grade'].unique()), palette='deep')
sns.barplot(x='grade', y='id', data=dataset[dataset['Default'] == 1], 
            estimator=lambda x: len(x) / nLate * 100,
            ax=ax3, order=sorted(dataset['grade'].unique()), palette='deep')

ax1.set_title('Overall')
ax2.set_title('No Default')
ax3.set_title('Default')
ax1.set_ylabel('Percentage')
ax2.set_ylabel('')
ax3.set_ylabel('')

plt.tight_layout()
plt.show()


# The grade of the loan is the companies estimate of the likelyhood of default for the loan. As should probably be expected the best graded loans (A and B) have a higher percentage of loans with no default than with a default. C is approximately the same percentage across no default and default and the worst graded loans (D, E, F and G) have a higher percentage of loans with default than with no default.

# #### Loan Amount

# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 3))

sns.distplot(dataset[dataset['Default'] == 0]['loan_amnt'], bins=40, ax=ax1, kde=False)
sns.distplot(dataset[dataset['Default'] == 1]['loan_amnt'], bins=40, ax=ax2, kde=False)

ax1.set_title('No Default')
ax2.set_title('Default')

ax1.set_xbound(lower=0)
ax2.set_xbound(lower=0)

plt.tight_layout()
plt.show()


# In[ ]:


ax1 = sns.violinplot(x='Default', y='loan_amnt', data=dataset)
ax1.set_ybound(lower=0)
plt.show()


# Both No default and default have a resonably similar distribution of loan the loan amount

# #### Loan Term

# In[ ]:


nNoLate = len(dataset[dataset['Default'] == 0])
nLate = len(dataset[dataset['Default'] == 1])

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(9, 3))

sns.barplot(x='term', y='id', data=dataset, 
            estimator=lambda x: len(x) / (nLate + nNoLate) * 100,
            ax=ax1, order=sorted(dataset['term'].unique()), palette='deep')
sns.barplot(x='term', y='id', data=dataset[dataset['Default'] == 0], 
            estimator=lambda x: len(x) / nNoLate * 100,
            ax=ax2, order=sorted(dataset['term'].unique()), palette='deep')
sns.barplot(x='term', y='id', data=dataset[dataset['Default'] == 1], 
            estimator=lambda x: len(x) / nLate * 100,
            ax=ax3, order=sorted(dataset['term'].unique()), palette='deep')

ax1.set_title('Overall')
ax2.set_title('No Default')
ax3.set_title('Default')
ax1.set_ylabel('Percentage')
ax2.set_ylabel('')
ax3.set_ylabel('')

plt.tight_layout()
plt.show()


# The longer term loans (60 months) make up a higher percentage of the defaults than the non defaulting loans.

# #### Interest Rate

# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 3))

sns.distplot(dataset[dataset['Default'] == 0]['int_rate'], bins=30, ax=ax1, kde=False)
sns.distplot(dataset[dataset['Default'] == 1]['int_rate'], bins=30, ax=ax2, kde=False)

ax1.set_title('No Default')
ax2.set_title('Default')

ax1.set_xbound(lower=0)
ax2.set_xbound(lower=0)

plt.tight_layout()
plt.show()


# In[ ]:


ax1 = sns.boxplot(x='Default', y='int_rate', data=dataset)
ax1.set_ybound(lower=0)
plt.show()


# The defaulting loans have a higher interest rate than non defaulting loans.

# In[ ]:


ax1 = sns.boxplot(x='grade', y='int_rate', data=dataset, hue='Default', 
                     order=sorted(dataset['grade'].unique()))
ax1.set_ybound(lower=0)
plt.show()


# Even controlling for the grade of the loan (as this will be used to calculate the interest rate) the defaulting loans still have a higher interest rate than non defaulting loans until you get to grades F and G

# #### Annual Income

# In[ ]:


ax1 = sns.boxplot(x='Default', y='annual_inc', data=dataset)
ax1.set_ybound(lower=0)
ax1.set_yscale('log')

plt.show()


# Defaulting loans have a lower annual income than the non defaulting loans. There are very few joint applications so we won't create graphs for joint income as well for these but we will may these variables when creating models.

# #### Debt to Income Ratio

# In[ ]:


ax1 = sns.boxplot(x='Default', y='dti', data=dataset)
ax1.set_ybound(lower=0, upper=50)
plt.show()


# Defaulting loans have a higher DTI

# #### Home Ownership

# In[ ]:


nNoLate = len(dataset[dataset['Default'] == 0])
nLate = len(dataset[dataset['Default'] == 1])

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(9, 3))

sns.barplot(x='home_ownership', y='id', data=dataset, 
            estimator=lambda x: len(x) / (nLate + nNoLate) * 100,
            ax=ax1, order=['MORTGAGE', 'OWN', 'RENT'], palette='deep')
sns.barplot(x='home_ownership', y='id', data=dataset[dataset['Default'] == 0], 
            estimator=lambda x: len(x) / nNoLate * 100,
            ax=ax2, order=['MORTGAGE', 'OWN', 'RENT'], palette='deep')
sns.barplot(x='home_ownership', y='id', data=dataset[dataset['Default'] == 1], 
            estimator=lambda x: len(x) / nLate * 100,
            ax=ax3, order=['MORTGAGE', 'OWN', 'RENT'], palette='deep')

ax1.set_title('Overall')
ax2.set_title('No Default')
ax3.set_title('Default')
ax1.set_ylabel('Percentage')
ax2.set_ylabel('')
ax3.set_ylabel('')

plt.tight_layout()
plt.show()


# Renters make up a higher percentage of defaults than non defaults.

# #### Length of Employment

# In[ ]:


nNoLate = len(dataset[dataset['Default'] == 0])
nLate = len(dataset[dataset['Default'] == 1])

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(9, 3))

sns.barplot(x='Emp_Length_Years', y='id', data=dataset, 
            estimator=lambda x: len(x) / (nLate + nNoLate) * 100,
            ax=ax1, palette='deep')
sns.barplot(x='Emp_Length_Years', y='id', data=dataset[dataset['Default'] == 0], 
            estimator=lambda x: len(x) / nNoLate * 100,
            ax=ax2, palette='deep')
sns.barplot(x='Emp_Length_Years', y='id', data=dataset[dataset['Default'] == 1], 
            estimator=lambda x: len(x) / nLate * 100,
            ax=ax3, palette='deep')

ax1.set_title('Overall')
ax2.set_title('No Default')
ax3.set_title('Default')
ax1.set_ylabel('Percentage')
ax2.set_ylabel('')
ax3.set_ylabel('')

plt.tight_layout()
plt.show()


# Employees that have been at a company 10+ years have lower percentage of the total Defaults than the No Defaults

# #### Earliest Line of Credit

# In[ ]:


ax1 = sns.boxplot(x='Default', y='Earliest_Cr_Line_Yr', data=dataset)
plt.show()

