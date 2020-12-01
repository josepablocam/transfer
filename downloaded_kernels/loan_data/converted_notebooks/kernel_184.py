
# coding: utf-8

# Understand the different data patterns in the lending data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import style

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_loan = pd.read_csv('../input/loan.csv', low_memory=False)


# In[ ]:


df_loan.shape


# In[ ]:


df_loan.columns


# In[ ]:


df_loan.describe()


# In[ ]:


df_loan.head(5)


# In[ ]:


df_loan.isnull().sum()

