
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbs

df = pd.read_csv("../input/loan.csv", low_memory=False)


# # Exploring Lending Club Loan Data
# 
# There are a lot of columns here, 74 in all. 
# Not everything will be interesting at the first pass. Let's look at the ones that seem to stand well on their own.
# About the loans:
# 
# * How much are the loans for?
# * How long are the loans taken for?
# * What is the interest rate?
# 
# About the peopl:
# 
# * What kind of people take out loans?
# * Where are they from?

# In[ ]:


df["addr_state"].value_counts().plot(kind='bar')


# In[ ]:


df["annual_inc"].plot.density()


# In[ ]:


df["funded_amnt"].hist()
plt.title('Funded Amount')

