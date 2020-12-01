
# coding: utf-8

#  - In this notebook, I used Python and SQL to check/visualize how some
#    variables affect interest rates. There are three steps:
#  - 1.Get to know the data
#  - 2.Clean data
#  - 3.Analyze data

# In[ ]:


import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


con = sqlite3.connect('../input/database.sqlite')
loan = pd.read_csv('../input/loan.csv')


# **1. Get to know the data**

# In[ ]:


loan.dtypes


# In[ ]:


loan.head()


# In[ ]:


plt.rc("figure", figsize=(6, 4))
loan["loan_amnt"].hist()
plt.title("distribution of loan amount")


# In[ ]:


plt.rc("figure", figsize=(6, 4))
loan["int_rate"].hist()
plt.title("distribution of interest rate")


# **2. Clean data**

# **a. explore 'loan_amnt' and 'annual_inc'**

# I want to explore how interest rate depends on loan amount and annual income, whose types are float. So I need to get some basic statistics.

# In[ ]:


loan[["loan_amnt","annual_inc"]].dropna().describe()


# OK, so I will divide loan_amnt to four category: low, medium-low, medium-high and high loan_amnt, and also divide annual_income to four category: low, medium-low, medium-high and high annual_inc.

# **b. extract needed data from the database**

# In[ ]:


loan_rate_related = pd.read_sql_query( """
SELECT loan_amnt, term, int_rate, grade, emp_title, emp_length, home_ownership, annual_inc,issue_d,
purpose, title, addr_state,application_type,
CASE WHEN loan_amnt < 8000 THEN 'low' 
     WHEN loan_amnt >= 8000 AND loan_amnt < 13000 THEN 'medium-low'
     WHEN loan_amnt >= 13000 AND loan_amnt < 20000 THEN 'medium-high'
     WHEN loan_amnt >= 20000 THEN 'high' END as loan_amnt_level,
CASE WHEN annual_inc < 45000 THEN 'low'
     WHEN annual_inc >= 45000 AND annual_inc <65000 THEN 'medium-low'
     WHEN annual_inc >= 65000 AND annual_inc < 90000 THEN 'medium-high'
     WHEN annual_inc >= 90000 THEN 'high' END as annual_inc_level
FROM loan
""",con)


# OK, this is the table(dataframe) containing all the information I am interested in.

# In[ ]:


loan_rate_related.head()


# In[ ]:


loan_rate_related.shape


# **c. Deal with NULL values**

# In[ ]:


loan_rate_related.isnull().sum()


# In[ ]:


loan_rate_related = loan_rate_related.dropna(subset=["loan_amnt","term","int_rate","grade","emp_length","home_ownership","annual_inc",
                                              "issue_d","purpose","addr_state","application_type"])


# In[ ]:


loan_rate_related.shape


# **d. Deal with data types**

# In[ ]:


loan_rate_related.dtypes


# convert int_rate to float

# In[ ]:


loan_rate_related["int_rate"]=loan_rate_related["int_rate"].apply(lambda x: float(x.rstrip("%")))


# **3. Analyze data**

#  - Next, I will explore how all variables (loan amount, term, grade,
#    employee length, home ownership, annual income, issue day, purpose,
#    state, application type) affect interest rate.
#  - It turns out that loan amount, employee length, annual income, home ownership,state and issue month do not affect the interest rate much.
#  - the term, grade, purpose and application type would affect the interest rate to some extent.

# In[ ]:


order = ["low", "medium-low","medium-high","high"]
sns.boxplot(x='loan_amnt_level',y="int_rate",data = loan_rate_related,order=order)
plt.title("how 'loan amount' affects 'interest rate' ")


# In[ ]:


plt.rc("figure", figsize=(6, 4))
sns.boxplot(x='term',y="int_rate",data = loan_rate_related)
plt.title("how 'term' affects 'interest rate'")


# In[ ]:


plt.rc("figure", figsize=(6, 4))
sns.boxplot(x='grade',y="int_rate",data = loan_rate_related,order=["A","B","C","D","E","F","G"])
plt.title("how 'grade' affects 'interest rate'")


# In[ ]:


loan["emp_title"].unique()


# It turns out that there are a lot of employee titles, so I'll skip this one.

# In[ ]:


loan["emp_length"].unique()


# In[ ]:


order = ['1 year', '2 years', '3 years', '4 years',
       '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years', 'n/a']
plt.rc("figure", figsize=(6, 4))
sns.boxplot(x='emp_length',y="int_rate",data = loan_rate_related,order=order)
plt.title("how 'employee length' affects 'interest rate'")
plt.xticks(size = 10,rotation = 80)


# In[ ]:


sns.boxplot(x='annual_inc_level',y="int_rate",data = loan_rate_related)
plt.title("how 'annual income' affects 'interest rate'")


# In[ ]:


plt.rc("figure", figsize=(6, 4))
sns.boxplot(x='home_ownership',y="int_rate",data = loan_rate_related)
plt.title("how 'home_ownership' affects 'int_rate'")


# In[ ]:


loan_rate_related["issue_d"].unique()


# I'd like to split issue days into issue months and issue years.

# In[ ]:


loan_rate_related["issue_d"] = loan_rate_related["issue_d"].str.split("-")


# In[ ]:


loan_rate_related["issue_month"] = loan_rate_related["issue_d"].str[0]


# In[ ]:


loan_rate_related["issue_year"] = loan_rate_related["issue_d"].str[1]


# In[ ]:


order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
sns.boxplot(x='issue_month',y="int_rate",data = loan_rate_related,order = order)
plt.title("how 'issu_month' affects 'interest rate'")


# In[ ]:


order = np.sort(loan_rate_related["issue_year"].unique().tolist())
sns.boxplot(x='issue_year',y="int_rate",data = loan_rate_related, order = order)
plt.title("how 'issue_year' affects 'interest rate'")


# In[ ]:


rate_by_purpose = pd.read_sql_query( """
SELECT purpose, avg(int_rate) AS avg_rate
FROM loan
GROUP BY purpose
ORDER BY avg_rate desc
""",con)
rate_by_purpose


# In[ ]:


order = rate_by_purpose["purpose"].tolist()
sns.boxplot(x='purpose',y="int_rate",data = loan_rate_related, order = order)
plt.xticks(size = 10,rotation = 80)
plt.title("how 'purpose' affects 'interest rate'")


# In[ ]:


rate_by_state = pd.read_sql_query( """
SELECT addr_state, avg(int_rate) AS avg_rate
FROM loan
GROUP BY addr_state
ORDER BY avg_rate desc
""",con)
rate_by_state


# In[ ]:


plt.rc("figure", figsize=(9, 4))
order = rate_by_state["addr_state"].tolist()
sns.boxplot(x='addr_state',y="int_rate",data = loan_rate_related, order = order)
plt.xticks(size = 10,rotation = 80)
plt.title("how 'state' affects 'interest rate'")


# In[ ]:


plt.rc("figure", figsize=(6, 4))
sns.boxplot(x='application_type',y="int_rate",data = loan_rate_related)
plt.title("how 'application type' affects 'interest rate'")


#  - It turns out that loan amount, employee length, annual income, home ownership,state and issue month do not affect the interest rate much.
#  - the term, grade, purpose and application type would affect the interest rate to some extent.
#  - Conclusion: It's very likely that you can get low interest rate if the term is 36 months, the grade is low, the purpose is one of educational, car or credit card, the state is Idaho, and the type is "individual"!
