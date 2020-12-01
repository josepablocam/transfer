#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print((check_output(["ls", "../input"]).decode("utf8")))

# Any results you write to the current directory are saved as output.


# In[2]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_rows", 20)
df = pd.read_csv("../input/cwurData.csv")
df


# In[3]:


df[df["country"] == "Estonia"]


# In[4]:


df.groupby("country")["quality_of_education"].mean()


# In[5]:


df2= pd.DataFrame(df.groupby("country")["quality_of_education"].mean())
df2.sort_values("quality_of_education", ascending = False)


# In[6]:


df3 = df[df["year"] == 2015]
df3.country.value_counts()


# Näitab palju keskmiselt patente ülikoolidel on kohtade järgi.

# In[9]:


df.patents.plot.hist(bins=20, grid=False, rwidth=0.80);


# Näitab, kuidas ülikoolide skoori ja publikatsioonide suhe on, ehk kas paremad ülikoolid väljastavad rohkem artikleid.

# In[10]:


df.plot.scatter("publications", "score", alpha=0.3);

