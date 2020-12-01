
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('../input/loan.csv',low_memory=False)


# In[ ]:


df.shape


# In[ ]:


df.head(3)


# In[ ]:


x=df.isnull().sum().sort_values(ascending=False)


# In[ ]:


df =df.drop(['dti_joint',
'verification_status_joint',
'annual_inc_joint',
'il_util',
'mths_since_rcnt_il',
'all_util',
'max_bal_bc',
'open_rv_24m',
'open_rv_12m',
'total_cu_tl',
'total_bal_il',
'open_il_24m',
'open_il_12m',
'open_il_6m',
'open_acc_6m',
'inq_fi',
'inq_last_12m',
'desc',
'mths_since_last_record',
'mths_since_last_major_derog',
'mths_since_last_delinq',
'next_pymnt_d'
],axis=1)


# In[ ]:


df.isnull().sum().sort_values(ascending=False)


# In[ ]:


emp_title=df['emp_title']


# In[ ]:


df=df.drop(['emp_title','total_rev_hi_lim','tot_coll_amt','tot_cur_bal'],axis=1)


# In[ ]:


df.isnull().sum().sort_values(ascending=False)


# In[ ]:


df.head(4)


# In[ ]:


sns.countplot(df['application_type'])


# In[ ]:


sns.countplot(df['grade'])


# In[ ]:


sns.countplot(df['term'])


# In[ ]:


corr=df.corr()
corr = (corr)
plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws=

{'size': 15},
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.title('Heatmap of Correlation Matrix')


# In[ ]:


loan_status_distribution=pd.DataFrame(df['loan_status'].value_counts())


# In[ ]:


loan_status_distribution=pd.DataFrame(df['loan_status'].value_counts())
loan_status_distribution.reset_index(inplace=True)
loan_status_distribution.columns=['Loan_Status_Category','Number of applicants']
loan_status_distribution


# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(10,10))
sns.countplot(x='loan_status',data=df,palette='cubehelix')
sns.despine(top=True,right=True)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


df.head(5)


# In[ ]:


df2=df.copy()
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
state_average_int_rate=df.groupby('addr_state').agg({'int_rate':np.average,'id':np.count_nonzero,'annual_inc':np.average})
state_average_int_rate.reset_index(inplace=True)
state_average_int_rate['id']=state_average_int_rate['id'].astype(str)
state_average_int_rate['interest']=state_average_int_rate['int_rate']
state_average_int_rate['int_rate']= 'Average Interest Rate: ' + state_average_int_rate['int_rate'].apply(lambda x: str(round(x,2)))+ "%"
state_average_int_rate['annual_inc']=(state_average_int_rate['annual_inc']/1000.0)
state_average_int_rate['annual_inc']=state_average_int_rate['annual_inc'].apply(lambda x: str(round(x,2)))
state_average_int_rate['text']='Number of Applicants: ' + state_average_int_rate['id']+'<br>'+ 'Average Annual Inc: $'+ state_average_int_rate['annual_inc']+'k'

scl= [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]]

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = state_average_int_rate['addr_state'],
        z = state_average_int_rate['interest'].astype(float),
        text=state_average_int_rate['text'],
        locationmode = 'USA-states',
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Interest Rates")
        ) ]

layout = dict(
        title = '<b>Interest Rate by US States</b><br>Additional Details: <br> Avreage Annual Inc \t Number of Applicants',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
        
             ))
    
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )


# In[ ]:


df.head(3)


# In[ ]:


df.columns

