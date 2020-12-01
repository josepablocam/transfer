#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


df = pd.read_csv("../input/cwurData.csv")


# In[ ]:


df


# In[ ]:


df[df["country"] == "Estonia"]


# In[ ]:


pd.DataFrame(df.groupby("country")["quality_of_education"].mean()).sort_values("quality_of_education", ascending=False)


# In[ ]:


df.groupby("country").size()


# In[ ]:


df[df["year"] == 2015].groupby("country").size()

