#!/usr/bin/env python
# coding: utf-8

# **Teeme läbi väikesed harjutused, et hiljem oleks lihtsam kodutööd teha.**
# 
# 

# In[ ]:


import numpy as np
import pandas as pd 

df = pd.read_csv("../input/cwurData.csv")


# 1) Leia kaggle’st dataset ‘World University Rankings’
# 
# 2) Tee uus kernel (notebook)
# 
# 3) Loe andmed dataseti failist ‘cwurData.csv’
# 
# 4) Kuva andmed tabelina

# In[ ]:


df


# 5) Kuva tabelist read, mis käivad Eesti ülikoolide kohta

# In[ ]:


df.loc[df["country"] == "Estonia"]


# 6) Kuva keskmine hariduse kvaliteedi näitaja grupeerituna riikide kaupa

# In[ ]:


quality_of_edu_mean = pd.DataFrame(df.groupby('country').quality_of_education.mean())
quality_of_edu_mean


# 7) Järjesta saadud andmed keskmise hariduse kvaliteedi näitaja järgi kahanevalt
# 
# Vihjed: Pane eelmise ülesande andmed uude DataFrame ning sorteeri uus DataFrame

# In[ ]:


quality_of_edu_mean.sort_values('quality_of_education', ascending=False)


# 8) Leida mitu korda iga riigi ülikoole tabelis esineb
# 

# In[ ]:


uni_frequency = pd.DataFrame(df.groupby('country').size())
uni_frequency.rename(index=str, columns={0:"frequency"}, inplace=True)
uni_frequency.sort_values("frequency", ascending=False)


# 8) a) Leida täpsemalt ainult 2015. aasta tulemuste kohta

# In[ ]:


uni_frequency_2015 = pd.DataFrame(df[df.year == 2015].groupby('country').size())
uni_frequency_2015.rename(index=str, columns={0:"frequency"}, inplace=True)
uni_frequency_2015.sort_values("frequency", ascending=False)


# 9) Mitu ülikooli on välja andnud n publikatsiooni.

# In[ ]:


df["publications"].plot.hist(title="N publikatsiooni välja andnud ülikoolide arv", rwidth=0.9, grid=True, color="m");


# 10)  Kuidas on seotud ülikoolide välja antud publikatsioonide arv tsiteerimiste arvuga.

# In[ ]:


publications_mean = pd.DataFrame(df.groupby('institution').publications.mean())
citations_mean = pd.DataFrame(df.groupby('institution').citations.mean())
info = np.array([publications_mean["publications"], citations_mean["citations"]])

scatter_table = pd.DataFrame(data=info[0:], index=["publications", "citations"]).transpose()
scatter_table.plot.scatter("publications","citations", marker=".", alpha=0.3, color="m");


# 
