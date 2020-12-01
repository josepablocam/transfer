#!/usr/bin/env python
# coding: utf-8

# ## Find the ranking of your University or dreamed University here and know what can be done to be a better University.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
data = pd.read_csv('../input/cwurData.csv')


# In[ ]:


data.head(3)


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


dt_country = data.country.value_counts()

fig = plt.figure(figsize=(6, 14))
dt_country.plot(kind='barh')
plt.title('Total Institute by Country')
plt.xlabel('Counts')
plt.show()


# In[ ]:


def plot_correlation(df,fg_width=9, fg_height=9):
    
    corr = df.corr()
    fig = plt.figure(figsize=(fg_width, fg_height))
    ax = fig.add_subplot(211)
    cax = ax.matshow(corr, cmap='Blues', interpolation='nearest')
    
    fig.colorbar(cax)
    plt.xticks(list(range(len(corr.columns))), corr.columns)
    plt.yticks(list(range(len(corr.columns))), corr.columns)

plot_correlation(data.drop(['score', 'national_rank', 'year'], axis=1))


# In[ ]:


dt_sub2 = data.ix[:,["world_rank", "publications","influence","citations", "broad_impact"]]
pd.scatter_matrix(dt_sub2, alpha=0.3, diagonal='kde', figsize=(9,9))
plt.show()


# In[ ]:


data[['world_rank','institution', 'country', 'year']][(data['world_rank']<=10)]


# In[ ]:


data[['world_rank','institution', 'country', 'year']][(data['country']=='China')&(data['world_rank']<=500)]


# In[ ]:





# In[ ]:





# In[ ]:




