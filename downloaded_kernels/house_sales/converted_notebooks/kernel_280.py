#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/kc_house_data.csv")


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

g = sns.lmplot(x="sqft_living", y="price", hue="floors", data=df, size=7 )
g.set_axis_labels("Living Room Square Feet", "Price")


# In[ ]:


sns.set(style="white")
h = sns.PairGrid(df[["bedrooms", "price", "grade"]], diag_sharey=False)
h.map_lower(sns.kdeplot, cmap="Blues_d")
h.map_upper(plt.scatter)
h.map_diag(sns.kdeplot, lw=3)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


df.price.plot(kind="hist", bins=100)
sns.despine()


# In[ ]:


df.condition.plot(kind="hist", bins=5)


# In[ ]:


df.yr_built.plot(kind="hist")


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:




