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

# In[ ]:


import numpy as np
import pandas as pd 

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_rows', 10)

df = pd.read_csv("../input/cwurData.csv")

df


# 5) Kuva tabelist read, mis käivad Eesti ülikoolide kohta

# In[ ]:


df2 = df[df["country"] == "Estonia"]
df2 


# 6) Kuva keskmine hariduse kvaliteedi näitaja grupeerituna riikide kaupa

# In[ ]:


df.sort_values("quality_of_education")


# 7) Järjesta saadud andmed keskmise hariduse kvaliteedi näitaja järgi kahanevalt
# 
# Vihjed: Pane eelmise ülesande andmed uude DataFrame ning sorteeri uus DataFrame

# In[ ]:


df3 = df.sort_values("quality_of_education", ascending=False)
df3


# 8) Leida mitu korda iga riigi ülikoole tabelis esineb
# 

# In[ ]:



koolid = set(df["country"])
print("Erinevaid koole: " + str(len(koolid)))


# 
