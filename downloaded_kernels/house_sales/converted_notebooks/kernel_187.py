#!/usr/bin/env python
# coding: utf-8

# **Andmestiku tutvustus:**
# 
# Antud andmetöötlus annab lühikese ülevaate kinnisvara olukorrast Kingi maakonnas Washingtonis aastatel 1900–2015. Kasutatud on „House Sales in King Country, USA“ andmebaasi ja sellest järgnevalt välja toodud oluliseim statistika.
# Maakonna halduskeskus ja suurim linn on Seattle. Maakonna pindala on 5975 km². Elanike arv oli 2010. aasta rahvaloenduse järgi 1 931 249. 
# 

# In[10]:


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


# **Andmete lugemine:**
# 
# Töölehe esimeses plokis teeme kõik vajaminevad impordid ning vajalikud seadistused, omistame tähekominisatsioonile „df“ kogu andmebaasi ja kutsume siis selle järgnevalt tabelina välja, mis näitab andmehulga veerupealkirju ning esimesi ja viimaseid ridu.
# DataFrame'i täpsemat ülesehitust saab uurida meetodiga info(). Järgmisena välja kutsutu len-käsklus näitab, et meie andmebaasis on 21613 rida.
# 

# In[11]:


df = pd.read_csv("../input/kc_house_data.csv")


# In[12]:


df


# In[13]:


df.info()


# In[14]:


len(df)


# In[20]:


print("Kõrgeim hind", df["price"].max(), "eurot")
print("Minimaalne hind:", df["price"].min(), "eurot")


# **Grupeering:**
# 
# Tehtud tabel analüüsib kahte näitajat: kinnisvara seisukorda ja kinnisvarale antud hinnet, mis on põhineb Kingi maakonna hindamisskaalal. Mõlemas kategoorias toob tabel välja miinimum (min) ja maksimum(max) näitaja, ning kolmas veerg näitab andmete keskmist. Tabelist võib välja lugeda, et viimastel aastatel ehitatud majade hinne on keskmiselt kõrgem kui 20. sajandi alguses rajatud kinnisvaral. Seisukorra keskmine on aga suurem hoopis vanematel majadel.

# In[16]:


df.groupby("yr_built").aggregate({"condition": [ "min", "max", "mean"],
                                      "grade" : [ "min", "max", "mean"]})


# **Histogramm:**
# 
# Antud histogramm näitab eri aastatel ehitutud majade sagedust. Enim maju ehitati Kingsis viimasel kümnendil ning kõige vähem ajavahemikul 1930-1940. Vastavalt siis umbes 3250 ning 700 elamut.
# 

# In[17]:


df.yr_built.plot.hist(bins=11, grid=False, rwidth=0.95, color="r", alpha = 0.3);


# **Hajuvusdiagramm: **
# 
# Graafik näitab elamispinna seost hinnaga. Üldiselt mida suurem on elamispind, seda kõrgem hind.
# 

# In[18]:


df.plot.scatter("price", "sqft_living", alpha=0.6, color = "turquoise");


# In[ ]:




