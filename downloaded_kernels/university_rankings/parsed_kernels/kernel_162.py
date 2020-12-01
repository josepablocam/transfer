#!/usr/bin/env python
# coding: utf-8

# **Teeme läbi väikesed harjutused, et hiljem oleks lihtsam kodutööd teha.**
# 
# 

# 1) Leia kaggle’st dataset ‘World University Rankings’
# 
# 2) Tee uus kernel (notebook)
# 
# 3) Loe andmed dataseti failist ‘cwurData.csv’
# 
# 4) Kuva andmed tabelina

# In[1]:


import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('../input/cwurData.csv')

data


# 5) Kuva tabelist read, mis käivad Eesti ülikoolide kohta

# In[2]:


data[data.country == 'Estonia']


# 6) Kuva keskmine hariduse kvaliteedi näitaja grupeerituna riikide kaupa

# In[3]:


c = pd.DataFrame(data.groupby('country').quality_of_faculty.mean())

c


# 7) Järjesta saadud andmed keskmise hariduse kvaliteedi näitaja järgi kahanevalt
# 
# Vihjed: Pane eelmise ülesande andmed uude DataFrame ning sorteeri uus DataFrame

# In[4]:


c.sort_values('quality_of_faculty', ascending=False)


# 8) Leida mitu korda iga riigi ülikoole tabelis esineb
#         a) Leida täpsemalt ainult 2015. aasta tulemuste kohta
# 

# In[5]:


pd.DataFrame(data[data.year == 2015].groupby('country').size())


# 
