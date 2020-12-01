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

# In[4]:


import numpy as np
import pandas as pd 

df = pd.read_csv("../input/cwurData.csv")
df


# 5) Kuva tabelist read, mis käivad Eesti ülikoolide kohta

# In[13]:


df[df["country"] == "Estonia"]


# 6) Kuva keskmine hariduse kvaliteedi näitaja grupeerituna riikide kaupa

# In[24]:


df.groupby(["country"])["quality_of_education"].mean().round(1).sort_values(ascending=False)


# 7) Järjesta saadud andmed keskmise hariduse kvaliteedi näitaja järgi kahanevalt
# 
# Vihjed: Pane eelmise ülesande andmed uude DataFrame ning sorteeri uus DataFrame

# 
# Tegin 6. ja 7. koos

# 8) Leida mitu korda iga riigi ülikoole tabelis esineb
#         a) Leida täpsemalt ainult 2015. aasta tulemuste kohta
# 

# In[33]:


tulemused = df[df["year"] == 2015]
tulemused
tulemused["country"].value_counts()


# 
