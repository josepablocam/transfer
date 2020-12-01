
# coding: utf-8

# #Loans Per Captia
# I've seen a lot of people who are exploring this data look at the raw number of loans per state.  It's interesting in that it fairly accurately shows the ranking of states by population.  So, in this short script, I look at the loans granted per captia.  I got the population data from [this](https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_population) Wikipedia page.

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/loan.csv', usecols = ['loan_amnt', 'addr_state'])


# In[ ]:


perStatedf = df.groupby('addr_state', as_index = False).count().sort_values(by = 'loan_amnt', ascending=False)
perStatedf.columns = ['State', 'Num_Loans']


# Here's the plot of the raw loan numbers by state.

# In[ ]:


fig, ax = plt.subplots(figsize = (16,8))
ax = sns.barplot(x='State', y='Num_Loans', data=perStatedf)
ax.set(ylabel = 'Number of Loans', title = 'Loans per State')
plt.show()


# I load the population data in as a dictionary, convert it to a dataframe and merge it with my other data.  I could have probably found an easier way to load in the population data without entering it in by hand, but I'm pretty good at ten key so it took less time than looking for the 'easier' way.

# In[ ]:


statePop = {'CA' : 39144818,
'TX' : 27469144,
'FL' : 20271878,
'NY' : 19795791,
'IL' : 12859995,
'PA' : 12802503,
'OH' : 11613423,
'GA' : 10214860,
'NC' : 10042802,
'MI' : 9922576,
'NJ' : 8958013,
'VA' : 8382993,
'WA' : 7170351,
'AZ' : 6828065,
'MA' : 6794422,
'IN' : 6619680,
'TN' : 6600299,
'MO' : 6083672,
'MD' : 6006401,
'WI' : 5771337,
'MN' : 5489594,
'CO' : 5456574,
'SC' : 4896146,
'AL' : 4858979,
'LA' : 4670724,
'KY' : 4425092,
'OR' : 4028977,
'OK' : 3911338,
'CT' : 3890886,
'IA' : 3123899,
'UT' : 2995919,
'MS' : 2992333,
'AK' : 2978204,
'KS' : 2911641,
'NV' : 2890845,
'NM' : 2085109,
'NE' : 1896190,
'WV' : 1844128,
'ID' : 1654930,
'HI' : 1431603,
'NH' : 1330608,
'ME' : 1329328,
'RI' : 1053298,
'MT' : 1032949,
'DE' : 945934,
'SD' : 858469,
'ND' : 756927,
'AK' : 738432,
'DC' : 672228,
'VT' : 626042,
'WY' : 586107}
statePopdf = pd.DataFrame.from_dict(statePop, orient = 'index').reset_index()
statePopdf.columns = ['State', 'Pop']
perStatedf = pd.merge(perStatedf, statePopdf, on=['State'], how = 'inner')


# In[ ]:


perStatedf['PerCaptia'] = perStatedf.Num_Loans / perStatedf.Pop
fig, ax = plt.subplots(figsize = (16,8))
ax = sns.barplot(x='State', y='PerCaptia', data=perStatedf.sort_values(by = 'PerCaptia', ascending=False))
ax.set(ylabel = 'Number of Loans', title = 'Per Captia Loans by State')
plt.show()


# Here we can see that per person, Nevada takes out the most loans by a fair margin.  The former leader, California, is now ranked at number 10.
# 
# Now, because I have the data right there, I'm going to look at loan amount by state and per capita loan amount by state.

# In[ ]:


perStatedf = df.groupby('addr_state', as_index = False).sum().sort_values(by = 'loan_amnt', ascending=False)
perStatedf.columns = ['State', 'loan_amt']


# In[ ]:


fig, ax = plt.subplots(figsize = (16,8))
ax = sns.barplot(x='State', y='loan_amt', data=perStatedf)
ax.set(ylabel = 'Number of Loans', title = 'Total Loan Amount per State')
plt.show()


# In[ ]:


perStatedf = pd.merge(perStatedf, statePopdf, on=['State'], how = 'inner')
perStatedf['PerCaptia'] = perStatedf.loan_amt / perStatedf.Pop


# In[ ]:


fig, ax = plt.subplots(figsize = (16,8))
ax = sns.barplot(x='State', y='PerCaptia', data=perStatedf.sort_values(by = 'PerCaptia', ascending=False))
ax.set(ylabel = 'Number of Loans', title = 'Per Captia Loan Amount by State')
plt.show()


# We can see again, that the raw loan amount by state follows the state populations pretty close.  Again, when you look at the per capita amounts, Nevada is at the top.  Here we see that the former number 1, California, again drops in rank.  It's now in thirteenth place.
# 
# Next, I'm going to look at the per capita bad loans.

# In[ ]:


df = pd.read_csv('../input/loan.csv', usecols = ['loan_status', 'addr_state'])
df.loan_status.unique()


# In[ ]:


badLoan  = ['Charged Off', 
            'Default', 
            'Late (31-120 days)', 
            'Late (16-30 days)', 'In Grace Period', 
            'Does not meet the credit policy. Status:Charged Off'] 


# In[ ]:


df['isBad'] = [ 1 if x in badLoan else 0 for x in df.loan_status]
perStatedf = df.groupby('addr_state', as_index = False).sum().sort_values(by = 'isBad', ascending=False)
perStatedf.columns = ['State', 'badLoans']


# In[ ]:


fig, ax = plt.subplots(figsize = (16,8))
ax = sns.barplot(x='State', y='badLoans', data=perStatedf)
ax.set(ylabel = 'Number of Loans', title = 'Total Bad Loans per State')
plt.show()


# In[ ]:


perStatedf = pd.merge(perStatedf, statePopdf, on=['State'], how = 'inner')
perStatedf['PerCaptia'] = perStatedf.badLoans / perStatedf.Pop


# In[ ]:


fig, ax = plt.subplots(figsize = (16,8))
ax = sns.barplot(x='State', y='PerCaptia', data=perStatedf.sort_values(by = 'PerCaptia', ascending=False))
ax.set(ylabel = 'Number of Loans', title = 'Per Captia Bad Loans by State')
plt.show()


# Again we see that Nevada tops the charts with the most per capita bad loans.  The most interesting result is Washington DC.  It is 5th in total loans per capita, but it is 30th in per capita bad loans.
# 
# Looking at these results,  I think looking at the percentage of bad loans by state would offer more insight into this.

# In[ ]:


df = pd.read_csv('../input/loan.csv', usecols = ['loan_status', 'addr_state'])
perStatedf = df.groupby('addr_state', as_index = False).count().sort_values(by = 'loan_status', ascending = False)
perStatedf.columns = ['State', 'totalLoans']
df['isBad'] = [ 1 if x in badLoan else 0 for x in df.loan_status]
badLoansdf = df.groupby('addr_state', as_index = False).sum().sort_values(by = 'isBad', ascending = False)
badLoansdf.columns = ['State', 'badLoans']
perStatedf = pd.merge(perStatedf, badLoansdf, on = ['State'], how = 'inner')
perStatedf['percentBadLoans'] = (perStatedf.badLoans / perStatedf.totalLoans)*100


# In[ ]:


fig, ax = plt.subplots(figsize = (16,8))
ax = sns.barplot(x='State', y='percentBadLoans', data=perStatedf.sort_values(by = 'percentBadLoans', ascending=False))
ax.set(ylabel = 'Percent', title = 'Percent of Bad Loans by State')
plt.show()


# In[ ]:


perStatedf.sort_values(by = 'percentBadLoans', ascending = False).head()


# With how small the sample from Iowa is, I think we should disregard it. That puts Nevada back on top.  This makes the numbers for Washington DC even more interesting though, as it drops all the way down to 47th place.
