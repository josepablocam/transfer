
# coding: utf-8

# ##### Source of Data:  https://www.kaggle.com/wendykan/lending-club-loan-data

# In[2]:


import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


# ### Metadata
# 
# - Risk_Score:	For applications prior to November 5, 2013 the risk score is the borrower's FICO score. For applications after November 5, 2013 the risk score is the borrower's Vantage score.
# - annual_inc:	The self-reported annual income provided by the borrower during registration.
# - annual_inc_joint:	The combined self-reported annual income provided by the co-borrowers during registration
# - application_type:	Indicates whether the loan is an individual application or a joint application with two co-borrowers
# - collections_12_mths_ex_med:	Number of collections in 12 months excluding medical collections
# - delinq_2yrs:	The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years
# - desc:	Loan description provided by the borrower
# - dti:	A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.
# - dti_joint:	A ratio calculated using the co-borrowers' total monthly payments on the total debt obligations, excluding mortgages and the requested LC loan, divided by the co-borrowers' combined self-reported monthly income
# - emp_length:	Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. 
# - emp_title:	The job title supplied by the Borrower when applying for the loan.*
# - fico_range_high:	The upper boundary range the borrower’s FICO at loan origination belongs to.
# - fico_range_low:	The lower boundary range the borrower’s FICO at loan origination belongs to.
# - funded_amnt:	The total amount committed to that loan at that point in time.
# - funded_amnt_inv:	The total amount committed by investors for that loan at that point in time.
# - grade:	Lending Club assigned loan grade
# - home_ownership:	The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.
# - initial_list_status:	The initial listing status of the loan. Possible values are – W, F
# - inq_last_6mths:	The number of inquiries in past 6 months (excluding auto and mortgage inquiries)
# - installment:	The monthly payment owed by the borrower if the loan originates.
# - int_rate:	Interest Rate on the loan
# - is_inc_v:	Indicates if income was verified by LC, not verified, or if the income source was verified
# - last_credit_pull_d:	The most recent month LC pulled credit for this loan
# - last_fico_range_high:	The upper boundary range the borrower’s last FICO pulled belongs to.
# - last_fico_range_low:	The lower boundary range the borrower’s last FICO pulled belongs to.
# - last_pymnt_amnt:	Last total payment amount received
# - loan_status:	Current status of the loan
# - mths_since_last_delinq:	The number of months since the borrower's last delinquency.
# - mths_since_last_major_derog:	Months since most recent 90-day or worse rating
# - mths_since_last_record:	The number of months since the last public record.
# - next_pymnt_d:	Next scheduled payment date
# - open_acc:	The number of open credit lines in the borrower's credit file.
# - out_prncp:	Remaining outstanding principal for total amount funded
# - out_prncp_inv:	Remaining outstanding principal for portion of total amount funded by investors
# - pub_rec:	Number of derogatory public records
# purpose	A category provided by the borrower for the loan request. 
# - pymnt_plan:	Indicates if a payment plan has been put in place for the loan
# - recoveries:	post charge off gross recovery
# - revol_bal:	Total credit revolving balance
# - revol_util:	Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.
# - term:	The number of payments on the loan. Values are in months and can be either 36 or 60.
# - total_acc:	The total number of credit lines currently in the borrower's credit file
# - total_pymnt:	Payments received to date for total amount funded
# - total_pymnt_inv:	Payments received to date for portion of total amount funded by investors
# - total_rec_int:	Interest received to date
# - total_rec_late_fee:	Late fees received to date
# - total_rec_prncp:	Principal received to date
# - verified_status_joint:	Indicates if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified
# - open_acc_6m:	Number of open trades in last 6 months
# - open_il_6m:	Number of currently active installment trades
# - open_il_12m:	Number of installment accounts opened in past 12 months
# - open_il_24m:	Number of installment accounts opened in past 24 months
# - mths_since_rcnt_il:	Months since most recent installment accounts opened
# - total_bal_il:	Total current balance of all installment accounts
# - il_util:	Ratio of total current balance to high credit/credit limit on all install acct
# - open_rv_12m:	Number of revolving trades opened in past 12 months
# - open_rv_24m:	Number of revolving trades opened in past 24 months
# - max_bal_bc:	Maximum current balance owed on all revolving accounts
# - all_util:	Balance to credit limit on all trades
# - total_rev_hi_lim:  	Total revolving high credit/credit limit
# - inq_fi:	Number of personal finance inquiries
# - total_cu_tl:	Number of finance trades
# - inq_last_12m:	Number of credit inquiries in past 12 months
# - acc_now_delinq:	The number of accounts on which the borrower is now delinquent.
# - tot_coll_amt:	Total collection amounts ever owed
# - tot_cur_bal:	Total current balance of all accounts

# In[3]:


df_loan = pd.read_csv("../input/loan.csv",low_memory=False)


# ## Part I: Data Exploration
# 
# In this section of our code, we will explore the data. Firstly, we need to understand what variables are included within the data.. what is the data? What variables can we compare against one another to learn more about the data?
# 
# This section will also include visualization of the data. This will help to assist our exploration and really dig deep and understand the underlying data trends that are not visible when viewing a dataframe. 
# 
# ### We will explore the following: 
# 
# ###### Loans Issued 
# 
# ###### Job Title
# - Number of loans issued per Job Title 
# - Job titles versus Defaulting
# - Job titles versus value of loan (Job titles granted most valuables loans may be riskiest)
# ###### Interest Rate 
# - Interest rate versus Grade
# - Interest rate versus Defaulting
# - Amount funded for Charged-off/Other Default Accounts?
# - Annual Income versus funded amount and interest rate
# ###### State of Residence 
# - State versus loans issued 
# - State versus default loans
# ###### Individual/Joint Loan
# - Loan type versus interest rate
# - Loan type versus default status
# 

# In[4]:


df_loan.describe()


# The dataset has 887379 rows × 74 columns. This is a lot of information and we do not necessarily need all of it so let's delete some columns that we know are completely useless to our analysis. 
# 
# ###### Explanation for dropping certain variables/columns 
# 
# subgrade - Extra information, we already have grade 
# 
# desc - too much text, we cannot convert this into anything useful
# 
# title - Same as 'purpose' but has more detail
# 
# zip_code - last few characters are not shown, not useful to us

# In[5]:


df_loan.drop(['url', 'desc', 'policy_code', 'sub_grade', 'member_id', 'title', 'open_rv_12m', 'open_rv_24m',
              'total_cu_tl', 'mths_since_rcnt_il', 'collection_recovery_fee', 'earliest_cr_line',
              'total_rec_late_fee', 'recoveries', 'next_pymnt_d', 'max_bal_bc',
             'total_rec_prncp', 'last_credit_pull_d','out_prncp_inv', 'out_prncp', 
              'acc_now_delinq', 'all_util', 'total_rev_hi_lim' ], axis = 1, inplace = True)


# In[6]:


#Checking number of columns dropped
df_loan.shape 


# We will only explore and analyze data from 2010-2015 (In order to minimize our data points)

# In[7]:


df_loan['issue_month'], df_loan['issue_year'] = df_loan['issue_d'].str.split('-', 1).str
df_loan['issue_date'] = df_loan['issue_d']
#Now we have columns, Issue_month, Issue_year, issue_date at the end of the DF


# In[8]:


# Run this function After
df_loan['issue_d'] = pd.to_datetime(df_loan['issue_d'])
df_loan['issue_d'].dtypes


# In[10]:


df_loan.index = df_loan['issue_d']
#del df_loan['issue_d']
df_loan_dt = df_loan['2010-01-01': '2015-12-01']
# DO NOT RUN THIS UNTIL CONFIRMED COLUMNS df_loan_dt.dropna
print ("before:", df_loan.shape)
print ("after:", df_loan_dt.shape)


# Seems like majority of our dataset is from those 5 years, so dropping three years did not make a significant difference on the number of rows

# Now let's plot the trend of loans granted throughout the years, Loan amount vs issue date

# In[11]:


plt.figure(figsize = (18,5))
g = sns.pointplot(x='issue_date', y='loan_amnt', 
                  data=df_loan_dt[df_loan_dt['issue_date'] >= '2009'],
                 order =['Jan-2010', 'Feb-2010', 'Mar-2010', 'Apr-2010', 'May-2010', 'Jun-2010',
                         'July-2010', 'Aug-2010', 'Sep-2010', 'Oct-2010', 'Nov-2010', 'Dec-2010',
                         'Jan-2011', 'Feb-2011', 'Mar-2011', 'Apr-2011', 'May-2011', 'Jun-2011',
                         'July-2011', 'Aug-2011', 'Sep-2011', 'Oct-2011', 'Nov-2011', 'Dec-2011',
                         'Jan-2012', 'Feb-2012', 'Mar-2012', 'Apr-2012', 'May-2012', 'Jun-2012',
                         'July-2012', 'Aug-2012', 'Sep-2012', 'Oct-2012', 'Nov-2012', 'Dec-2012',
                         'Jan-2013', 'Feb-2013', 'Mar-2013', 'Apr-2013', 'May-2013', 'Jun-2013',
                         'July-2013', 'Aug-2013', 'Sep-2013', 'Oct-2013', 'Nov-2013', 'Dec-2013',
                        'Jan-2014', 'Feb-2014', 'Mar-2014', 'Apr-2014', 'May-2014', 'Jun-2014',
                         'July-2014', 'Aug-2014', 'Sep-2014', 'Oct-2014', 'Nov-2014', 'Dec-2014',
                        'Jan-2015', 'Feb-2015', 'Mar-2015', 'Apr-2015', 'May-2015', 'Jun-2015',
                         'July-2015', 'Aug-2015', 'Sep-2015', 'Oct-2015', 'Nov-2015', 'Dec-2015'])
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel("Year", fontsize=5)
g.set_ylabel("Mean Loan Amount", fontsize=5)
g.set_title("Loan Amount (mean) issued by Year", fontsize=10)
plt.show()


# Seems as though the mean loan amount ($) has increased over the years, but this could be due to the fact that their clientelle is increasing. Let's plot how their clientelle has increased over the years. 

# In[12]:


df_count = df_loan_dt.groupby('issue_d')['id'].nunique()
df_count.head()


# Below plot shows the growth of the company overtime. Number of loans that have been issued in the past several years

# In[13]:


plt.figure(figsize = (18,5))
ax = sns.countplot(x='issue_d',
                  data=df_loan_dt[df_loan_dt['issue_year'] > '2009'],
                  order =['Jan-2010', 'Feb-2010', 'Mar-2010', 'Apr-2010', 'May-2010', 'Jun-2010',
                         'July-2010', 'Aug-2010', 'Sep-2010', 'Oct-2010', 'Nov-2010', 'Dec-2010',
                         'Jan-2011', 'Feb-2011', 'Mar-2011', 'Apr-2011', 'May-2011', 'Jun-2011',
                         'July-2011', 'Aug-2011', 'Sep-2011', 'Oct-2011', 'Nov-2011', 'Dec-2011',
                         'Jan-2012', 'Feb-2012', 'Mar-2012', 'Apr-2012', 'May-2012', 'Jun-2012',
                         'July-2012', 'Aug-2012', 'Sep-2012', 'Oct-2012', 'Nov-2012', 'Dec-2012',
                         'Jan-2013', 'Feb-2013', 'Mar-2013', 'Apr-2013', 'May-2013', 'Jun-2013',
                         'July-2013', 'Aug-2013', 'Sep-2013', 'Oct-2013', 'Nov-2013', 'Dec-2013',
                        'Jan-2014', 'Feb-2014', 'Mar-2014', 'Apr-2014', 'May-2014', 'Jun-2014',
                         'July-2014', 'Aug-2014', 'Sep-2014', 'Oct-2014', 'Nov-2014', 'Dec-2014',
                        'Jan-2015', 'Feb-2015', 'Mar-2015', 'Apr-2015', 'May-2015', 'Jun-2015',
                         'July-2015', 'Aug-2015', 'Sep-2015', 'Oct-2015', 'Nov-2015', 'Dec-2015'])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_ylabel("Count of Loans", fontsize=10)
ax.set_xlabel("Issue Date", fontsize=10)
ax.set_title("Lending Club Growth Overtime", fontsize =15)


# # Job Title
# #### - Number of loans issued per Employment Length ✔ - Job titles versus Defaulting - Job Length vs Loans ✔
# - what are top emp_titles for charged-off accounts? ✔

# In[14]:


# Need to combine RN and Registered Nurse labels as they mean the same thing. 
df_loan_dt['emp_title'] = df_loan_dt['emp_title'].replace({'RN':'Registered Nurse'})
df_loan_dt['emp_title'] = df_loan_dt['emp_title'].replace({'manager':'Manager'})
df_loan_dt['emp_title'] = df_loan_dt['emp_title'].replace({'driver':'Driver'})
df_loan_dt['emp_title'] = df_loan_dt['emp_title'].replace({'supervisor':'Supervisor'})
df_loan_dt['emp_title'] = df_loan_dt['emp_title'].replace({'owner':'Owner'})


# In[16]:


plt.figure(figsize=(10,5))
m = sns.countplot(x='emp_length',
                  data=df_loan_dt[df_loan_dt['issue_d'] > '2010-01-01'], 
                  order =['10+ years','9 years','8 years', '7 years', '6 years', '5 years', '4 years', '3 years',
                          '2 years','1 year', '<1 year'])
m.set_xticklabels(m.get_xticklabels(),rotation=45)
m.set_title("Loans VS Length of Employment", fontsize=10)
m.set_xlabel("Employment Length", fontsize=5)
m.set_ylabel("Loans", fontsize=5)


# In[19]:


from matplotlib import cm

temp_title = df_loan_dt.emp_title.value_counts()
temp_title1 = temp_title.head(10)
print (temp_title1)

plt.figure(figsize=(15,5))
colors = cm.Blues(temp_title1 / float(max(temp_title1)))
plot = plt.scatter(temp_title1, temp_title1, c = temp_title1, cmap = 'Blues')
plt.clf()
plt.colorbar(plot)
plt.bar(range(len(temp_title1)), temp_title1, color = colors)
plt.xticks(np.arange(10), ('Teacher', 'Manager', 'Registered Nurse', 'Owner',
                           'Supervisor', 'Sales','Project Manager','Driver',
                          'Office Manager', 'General Manager'), rotation=90)
plt.title("Top 10 Job Titles for Clients", fontsize=20)
plt.xlabel("Employment Title", fontsize=10)
plt.show()


# In[20]:


#Removed Current and Fully Paid as they were skewing the graph we were looking to create
#Graph would measure Emp title (top 10) versus current status 
df_temp=df_loan_dt[df_loan_dt.loan_status != 'Current']
df_temp=df_temp[df_temp.loan_status != 'Fully Paid']
df_temp.loan_status.unique()


# ###### Plotting the distribution of Funded Amounts ($)

# In[21]:


#convert "funded_amnt" into a numpy array and then use the plotly function to plot it as a distplot
#What is the mean for column "funded_amnt"? 14741
plt.figure(figsize=(15,5))
g = sns.distplot((df_loan_dt["funded_amnt"]))
g.set_xlabel("Funded Amount ($)", fontsize=12)
g.set_ylabel("Distribuition", fontsize=12)
g.set_title("Funded Amount($) distribuition", fontsize=20)


# In[23]:


plt.figure(figsize = (14,5))
#Looking the count of defaults though the issue_d that is The month which the loan was funded
g = sns.boxplot(x='loan_status', y="funded_amnt",
                   data=df_loan_dt)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Amount Funded ($)", fontsize=15)
g.set_title("Loan Status by Amount Funded", fontsize=20)
print (df_loan_dt.loan_status.unique())


# #### TOP EMPLOYEE TITLES FOR CHARGED OFF ACCOUNTS

# In[24]:


#Create DF for only Charged Off Accounts
df_CO = df_temp.loc[df_temp['loan_status'] == 'Charged Off']

emptitle_CO = df_CO.emp_title.value_counts()
emptitle_CO = emptitle_CO.head(10)

plt.figure(figsize=(15,5))
colors = cm.Blues(temp_title1 / float(max(temp_title1)))
plot = plt.scatter(temp_title1, temp_title1, c = temp_title1, cmap = 'Blues')
plt.clf()
plt.colorbar(plot)
plt.bar(range(len(temp_title1)), temp_title1, color = colors)
plt.xticks(np.arange(10), ('Manager','Teacher','Registered Nurse',
                            'Supervisor','Driver' ,'Owner','Sales', 'Project Manager','General Manager',
                          'Office Manager'), rotation=45)
plt.title("Top 10 Job Titles for Charged Off Accounts", fontsize=20)
plt.xlabel("Employment Title", fontsize=10)
plt.show()


# # Interest Rate 
# ###### - Interest rate versus Grade✔ - Interest rate versus Defaulting✔ - Amount funded for Charged-off/Other Default Accounts✔

# In[25]:


plt.figure(figsize = (14,5))
#Looking the count of defaults though the issue_d that is The month which the loan was funded
g = sns.boxplot(x='grade', y="int_rate",
                   data=df_loan_dt,
               order =['A','B','C','D','E', 'F', 'G'])
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_xlabel("Grade", fontsize=12)
g.set_ylabel("Interest Rate (%)", fontsize=15)
g.set_title("Interest Rate VS Grade", fontsize=20)


# In[26]:


plt.figure(figsize = (14,7))
plt.subplot(211)
g = sns.boxplot(x='grade', y="funded_amnt",
                   data=df_loan_dt,
               order =['A','B','C','D','E', 'F', 'G'])
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_xlabel("Grade", fontsize=12)
g.set_ylabel("Funded Amount ($)", fontsize=15)
g.set_title("Funded Amount VS Grade", fontsize=20)

plt.subplot(212)
g = sns.barplot(x='grade', y="funded_amnt",
                   data=df_loan_dt,
               order =['A','B','C','D','E', 'F', 'G'])
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_xlabel("Grade", fontsize=12)
g.set_ylabel("Funded Amount ($)", fontsize=15)


# This is very interesting. Those with low scores are usually granted a higher funded amount, typically at a higher interest rate as well. Let's pull the stats summaries for each letter grade

# In[28]:


#Let's summarize the first, middle and last Grades
df_A = df_loan_dt.loc[df_loan_dt['grade'] == 'A']
df_D = df_loan_dt.loc[df_loan_dt['grade'] == 'D']
df_G = df_loan_dt.loc[df_loan_dt['grade'] == 'G']
print ('Grade A, D, G Summary Stats')
print ("Min, Max, Mean Interest Rate for Grade A:")
print (df_A["int_rate"].min(),',', df_A["int_rate"].max(),',', df_A["int_rate"].mean())

print ("Min, Max, Mean Interest Rate for Grade D:")
print (df_D["int_rate"].min(),',', df_D["int_rate"].max(),',', df_D["int_rate"].mean())


print ("Min, Max, Mean Interest Rate for Grade G:")
print (df_G["int_rate"].min(),',', df_G["int_rate"].max(),',', df_G["int_rate"].mean())


# In[29]:


plt.figure(figsize = (10,5))
g = sns.violinplot(x='loan_status', y="int_rate",
                   data=df_loan_dt)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_ylabel("Interest Rate (%)", fontsize=15)
g.set_xlabel("", fontsize=15)
g.set_title("Interest Rate VS Status", fontsize=20)


# In[30]:


# Finding Correlation Between the Numerical Variables 
print (df_loan_dt['int_rate'].corr(df_loan_dt['funded_amnt']))
print (df_loan_dt['int_rate'].corr(df_loan_dt['annual_inc']))

