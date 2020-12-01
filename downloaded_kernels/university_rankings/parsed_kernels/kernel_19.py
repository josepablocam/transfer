#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
df = pd.read_csv("../input/timesData.csv")
from subprocess import check_output
print((check_output(["ls", "../input"]).decode("utf8")))

# Any results you write to the current directory are saved as output.


# In[ ]:


df.info()


# In[2]:


df.head()


# In[3]:


# Create variable with TRUE if year is equal to 2016
yr_2016 = df['year'] == 2016


# In[4]:


#for Nigeria
# Create variable with TRUE if country is Nigeria
Nigeria = df['country'] == "Nigeria"
# Select all cases where the country is Nigeria and year is equal to 2016
Nigeria = df[Nigeria & yr_2016]
Nigeria


# In[5]:


#for Ghana
# Create variable with TRUE if country is Ghana
Ghana = df['country'] == "Ghana"
#Select all cases where the country is Ghana and  year is equal to 2016
Ghana = df[Ghana & yr_2016]
Ghana


# In[6]:


#for Kenya
# Create variable with TRUE if country is Kenya
Kenya = df['country'] == "Kenya"
# Select all cases where the country is  and  year is equal to 2016
Kenya = df[Kenya & yr_2016]
Kenya


# In[7]:


#for South Africa
# Create variable with TRUE if country is South Africa
SA = df['country'] == "South Africa"
# Select all cases where the country is South Africa and  year is equal to 2016
SA = df[SA & yr_2016]
SA


# In[8]:


#for Egypt
# Create variable with TRUE if country is Egypt
Egypt = df['country'] == "Egypt"
# Select all cases where the country is Egypt and  year is equal to 2016
Egypt = df[Egypt & yr_2016]
Egypt


# In[9]:


#for Morocco
# Create variable with TRUE if country is Morocco
Morocco = df['country'] == "Morocco"
# Select all casess country is Morocco and  year is equal to 2016
Morocco = df[Morocco & yr_2016]
Morocco


# In[10]:


#for Uganda
# Create variable with TRUE if country is Uganda
Uganda = df['country'] == "Uganda"
# Select all cases country is Uganda and  year is equal to 2016
Uganda = df[Uganda & yr_2016]
Uganda


# In[11]:


frames = [Nigeria, Ghana, Kenya, Uganda, SA, Morocco, Egypt]


# In[12]:


result = pd.concat(frames)


# In[13]:


result


# In[14]:


#columns for visualizations
uni_name = result['university_name']
teaching = result['teaching']
research = result['research']
international_coll =  result['international']
citations =  result['citations']
stud_pop = result['num_students']


# In[15]:


stud_num = [x.replace(",", "") for x in stud_pop] #taking out the commas in student population data


# In[17]:


#plotting universities vs student population
universities = uni_name.values.tolist() #converting the data frame to a list
objects = universities
y_pos = np.arange(len(objects))


# In[18]:


#first plot-Student Population of Selected African Universities
plt.bar(y_pos, stud_num, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation=90)
plt.ylabel('Student Population')
plt.title('Plot 1 - 2016 Student Population of selected African Universities')
 
plt.show()


# In[19]:


universities


# In[21]:


#Plot 2-Group bar charts for Teaching, Research, International Collaborations and citations
# Setting the positions and width for the bars
pos = np.arange(len(universities))
width = 0.2


# In[22]:


# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))
y1 = teaching
y2 = research
y3 = international_coll
y4 = citations
# Create a bar with Teaching data,
# in position pos,
plt.bar(pos,y1, width,  alpha=0.5, color='b', label='Teaching')

# Create a bar with research data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos], y2, width, alpha=0.5, color='r', label='Research')

# Create a bar with international collaborations data,
# in position pos + some width buffer,
plt.bar([p + width*2 for p in pos], y3, width, alpha=0.5, color='g', label='International Collaborations')

# Create a bar with citations data,
# in position pos + some width buffer,
plt.bar([p + width*3 for p in pos], y4, width, alpha=0.5, color='y', label='Citations')

# Set the y axis label
ax.set_ylabel('Ranking Parameters')

# Set the chart's title
ax.set_title('Bar chart for different ranking parameters')

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(universities, rotation=90)

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*4)
plt.ylim([0, 100] )

# Adding the legend and showing the plot
plt.legend(['Teaching', 'Research', 'International Collaborations', 'Citations'], loc='upper left')
plt.grid()
plt.show()


# In[30]:


# pairwise correlation between columns, dropping certain columns
result.drop(['world_rank', 'income', 'total_score', 'year'], axis=1).corr(method='spearman').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)


# In[ ]:




