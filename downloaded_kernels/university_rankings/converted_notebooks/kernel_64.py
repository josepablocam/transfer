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

# In[2]:


import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_rows", 20)
df = pd.read_csv("../input/cwurData.csv")
df


# 5) Kuva tabelist read, mis käivad Eesti ülikoolide kohta

# In[3]:


df[df["country"] == "Estonia"]


# 6) Kuva keskmine hariduse kvaliteedi näitaja grupeerituna riikide kaupa

# In[4]:


df.groupby("country")["quality_of_education"].mean()


# 7) Järjesta saadud andmed keskmise hariduse kvaliteedi näitaja järgi kahanevalt
# 
# Vihjed: Pane eelmise ülesande andmed uude DataFrame ning sorteeri uus DataFrame

# In[5]:


df2= pd.DataFrame(df.groupby("country")["quality_of_education"].mean())
df2.sort_values("quality_of_education", ascending = False)


# 8) Leida mitu korda iga riigi ülikoole tabelis esineb
#         a) Leida täpsemalt ainult 2015. aasta tulemuste kohta
# 

# In[7]:


df3 = df[df["year"] == 2015]
df3.country.value_counts()


# 
