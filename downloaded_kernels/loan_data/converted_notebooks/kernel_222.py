
# coding: utf-8

# #Looking into Loans by State
# I stole this notebook from **[Omar Peña](https://www.kaggle.com/greyfaux)** and I wanted to adapt it to questions about my state of WA.

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import style
plt.style.use('ggplot')


# In[ ]:


data = pd.read_csv('../input/loan.csv', low_memory=False)


# In[ ]:


state_count = data.addr_state.value_counts()

state_count.plot(kind = 'bar',figsize=(16,8), title = 'Loans per State')


# In[ ]:


wa_data = data.loc[data.addr_state == 'WA']

wa_x = range(1, 19435)

wa_loan_amnt = wa_data.loan_amnt


# In[ ]:


plt.figure(figsize=(16, 10))
plt.scatter(wa_x, wa_loan_amnt)

plt.xlim(1,12888)
plt.ylim(0, 37500)

plt.ylabel("Loan Amount")
plt.title("Loan Size in Washington")

plt.show()


# In[ ]:


wa_loan_amnt.describe()


# In[ ]:


plt.figure(figsize=(16,8))

mu = 14885.040393
sigma = 8493.736887
num_bins = 300

n, bins, patches = plt.hist(wa_loan_amnt, num_bins, normed=1, facecolor='blue', alpha=0.7)
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')

plt.xlabel("Loan Amount")
plt.title("Loan Amount Distribution in Washington")
plt.show()

