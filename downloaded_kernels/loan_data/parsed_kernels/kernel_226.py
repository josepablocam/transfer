
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
import pandas as pd
loans = pd.read_csv("../input/loan.csv")


# #h
# 
# Redoing this, I was going in a direction I was not very excited about
# #h
# 
# #hh
# I will be exploring the "bigger picture" of what lending across the U.S Looks like.
# 
# 
# If you have a descent idea, let me know. As an Economist I find this fascinating.
# 
