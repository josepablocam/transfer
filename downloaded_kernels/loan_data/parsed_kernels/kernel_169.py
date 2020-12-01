
# coding: utf-8

# I tried to make this generic, so you can cut and paste whatever query you want, **after replacing the input files with the appropriate data source**.  For this example, I'm just working my way down the list of most popular SQLite kernels, and the next in line is the one identified below.

# **Source of query:**<br>
# [Total loans by state barplot](https://www.kaggle.com/bruessow/total-loans-by-state-histogram) by SvenBrüssow

# In[ ]:


dbname = 'database.sqlite'


# In[ ]:


import numpy as np 
import pandas as pd
import sqlite3

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


path = "../input/"  #Insert path here
database = path + dbname

conn = sqlite3.connect(database)

tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table';""", conn)
tables


# ### PASTE QUERY IN THE BLOCK BELOW

# In[ ]:


query = '''
SELECT addr_state,
REPLACE(SUBSTR(QUOTE(
    ZEROBLOB((COUNT(*)/1000+1)/2)
), 3, COUNT(*)/1000), '0', '*')
AS total_loans
FROM loan
GROUP BY addr_state
ORDER BY COUNT(*) DESC;
'''


# In[ ]:


result = pd.read_sql( query, conn )


# In[ ]:


result


# In[ ]:


result.to_csv("result.csv", index=False)

