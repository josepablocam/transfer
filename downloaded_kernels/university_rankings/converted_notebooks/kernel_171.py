#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


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


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv('../input/timesData.csv')


# In[ ]:


odf_cp1 = df.copy()
odf_cp1['international_students'].replace('',np.nan, inplace=True) #replace all empty celss with Null value
odf_cp1['international_students'] = odf_cp1['international_students'].str.replace('%', '')
odf_cp1['international_students'] = odf_cp1['international_students'].astype(np.float)
print ("The mean of international student percentage:"
       + str(odf_cp1['international_students'].mean())+"%")


# In[ ]:


print ("The university with highest percent of international students:")
odf_cp1[odf_cp1['international_students'] == odf_cp1['international_students'].max()]


# In[ ]:


print ("The universities with lowest percent of international students:")
odf_cp1[odf_cp1['international_students'] == odf_cp1['international_students'].min()].head()


# In[ ]:


print("The mean of student_staff_ratio is:")
df['student_staff_ratio'].mean()


# In[ ]:


print("The world_ranking of those universities with student_staff_ratio under 3.0 is:")

df[df['student_staff_ratio']<3].loc[:, ['student_staff_ratio',
                                        'world_rank', 'university_name']]


# In[ ]:


print ("""To solve: How do I show all at once without writing code for each year?""")
print ("\nThe amount of universities from each country in top 100 world_rank (per year)")

g = df[['country', 'year', 'university_name']][df['world_rank'].str.len() < 3 ]
g.groupby(['year', 'country']).count()


# In[ ]:


print ('Average university enrollment does not seem to be increasing')
odf_cp2 = df.copy()
odf_cp2['num_students'] = odf_cp2['num_students'].str.replace(',','')
odf_cp2['num_students'] = odf_cp2['num_students'].astype(np.float)
odf_cp2.groupby('year')['num_students'].mean()


# In[ ]:


"""Cleaning female_male_ratio column"""
odf_cp3 = df.copy()
odf_cp3 = odf_cp3[odf_cp3['female_male_ratio'].str.len()>0] #keep only cells that are not empty
odf_cp3['female_male_ratio'] = odf_cp3['female_male_ratio'].str.replace('-','0')
odf_cp3['female_male_ratio'] = odf_cp3['female_male_ratio']                            .str.split(':', expand=True)#'expand' returns a dataframe
                                                        #instead of a list
odf_cp3['female_male_ratio'] = odf_cp3['female_male_ratio']                            .str[0:2] #grabs first 2 characters of the string in cell
odf_cp3['female_male_ratio'] = odf_cp3['female_male_ratio'].astype(np.float)

print('The university with highest percentage of female students')
odf_cp3[odf_cp3['female_male_ratio']==odf_cp3['female_male_ratio'].max()]


# In[ ]:


print('The percentage of female students has not increased.')
odf_cp3.groupby('year')['female_male_ratio'].mean()


# In[ ]:


print ('There is no correlation between rank of university and student to staff ratio')
odf_cp5 = df.copy()

# convert world rank columns to float (where necessary)
f = lambda x: int((int(x.split('-')[0]) + int(x.split('-')[1])) / 2) if len(str(x).strip()) > 3 else x

odf_cp5['world_rank'] = odf_cp5['world_rank'].str.replace('=','').map(
    f).astype('float')

vis1 = sns.lmplot(data=odf_cp5, x='student_staff_ratio', y='world_rank',                   fit_reg=False, hue='year', size=7, aspect=1)


# In[ ]:


print('Correlation between university rank and score.')
odf_cp4 = df.copy()
odf_cp4 = odf_cp4[odf_cp4['total_score'].str.len()>1] #cell with values '-' will be dropped
odf_cp4['total_score'] = odf_cp4['total_score'].astype(np.float)
# convert world rank columns to float (where necessary)
f = lambda x: int((int(x.split('-')[0]) + int(x.split('-')[1])) / 2) if len(str(x).strip()) > 3 else x

odf_cp4['world_rank'] = odf_cp4['world_rank'].str.replace('=','').map(
    f).astype('float')
vis2 = sns.lmplot(data=odf_cp4, x='total_score', y='world_rank',                   fit_reg=False, hue='year', size=7, aspect=1)

