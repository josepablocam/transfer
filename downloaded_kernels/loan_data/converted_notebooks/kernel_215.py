
# coding: utf-8

# **Python Analysis**
# 
# Aim is to use this data set in ways that focus on manipulating dates and running calculations in Python.
# 
# Lending Club lets you buy a slice of a loan when it originates. You can also buy and sell these slices of current loans. When participating in any of these activities, lets try to optimize our return relative to risk. 
# 

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

date = ['issue_d', 'last_pymnt_d']
cols = ['issue_d', 'term', 'int_rate','loan_amnt', 'total_pymnt', 'last_pymnt_d','sub_grade','grade','loan_status']
loans = pd.read_csv("../input/loan.csv", low_memory=False,
     parse_dates=date, usecols = cols, infer_datetime_format=True)


# In[ ]:


#Won't include loans that are Current
#Find any loan that started at least 3 years ago if a 3 year loan and at least 5 if 5 year loan   
latest = loans['issue_d'].max()
finished_bool = ((loans['issue_d'] < latest - pd.DateOffset(years=3)) & (loans['term'] == ' 36 months')) | ((loans['issue_d'] < latest - pd.DateOffset(years=5)) & (loans['term'] == ' 60 months'))

finished_loans = loans.loc[finished_bool]


# In[ ]:


#ROI and Time Past
finished_loans['roi'] = ((finished_loans.total_pymnt / finished_loans.loan_amnt)-1)*100 


#Return per unit of risk - B combines return and lower risk
print(finished_loans.groupby(['grade'])['roi'].mean()/finished_loans.groupby(['grade'])['roi'].std())


# In[ ]:


y = finished_loans.groupby(['grade'])['roi'].mean()
x = finished_loans.groupby(['grade'])['roi'].std()
label = ["A","B","C","D","E","F","G"]
fig, ax = plt.subplots()
plt.scatter(x, y)
plt.axis([0,50,0,12])
ax.set_ylabel('Return')
ax.set_xlabel('Standard Deviation')
for i in range(len(label)):
    plt.annotate(
    s = label[i],
    xy = (x[i] + .5 , y[i])
)
 

