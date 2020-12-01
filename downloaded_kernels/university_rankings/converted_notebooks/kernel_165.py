#!/usr/bin/env python
# coding: utf-8

# In[ ]:


td = pd.read_csv("../input/timesData.csv")
td.tail()


# In[ ]:


td.head(5)


# In[ ]:


td.world_rank.unique()


# In[ ]:


int(td.world_rank[1:10])


# In[ ]:


td.hist()

