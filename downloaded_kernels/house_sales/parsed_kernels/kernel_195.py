#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pandas import read_csv
data = read_csv("../input/kc_house_data.csv")


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.isnull().any()


# In[ ]:


target = "price"
features = data.drop(target,1).columns


# In[ ]:


features_by_dtype = {}

for f in features:
    dtype = str(data[f].dtype)
    if dtype not in list(features_by_dtype.keys()):
        features_by_dtype[dtype] = [f]
    else:
        features_by_dtype[dtype] += [f]
        
for k in list(features_by_dtype.keys()):
    string = "%s: %s" % (k , len(features_by_dtype[k]))
    print(string)


# In[ ]:


keys = iter(list(features_by_dtype.keys()))


# In[ ]:


k = next(keys)
dtype_list = features_by_dtype[k]
for d in dtype_list:
    string = "%s: %s" % (d,len(data[d].unique()))
    print(string)


# In[ ]:


count_features = ["bedrooms"]


# In[ ]:


categorical_features = ["waterfront"]


# In[ ]:


count_features += ["view", "condition", "grade"]


# In[ ]:


categorical_features += ["zipcode"]


# In[ ]:


temporal_features = ["yr_renovated", "yr_built"]


# In[ ]:


numerical_features = [f for f in dtype_list if not f in categorical_features + temporal_features + ["id"]]


# In[ ]:


k = next(keys)
dtype_list = features_by_dtype[k]
for d in dtype_list:
    string = "%s: %s" % (d,len(data[d].unique()))
    print(string)


# In[ ]:


temporal_features += dtype_list


# In[ ]:


k = next(keys)
dtype_list = features_by_dtype[k]
for d in dtype_list:
    string = "%s: %s" % (d,len(data[d].unique()))
    print(string)


# In[ ]:


count_features += ["floors","bathrooms"]


# In[ ]:


numerical_features += dtype_list


# In[ ]:


numerical_features


# In[ ]:


count_features


# In[ ]:


categorical_features


# In[ ]:


temporal_features


# ---

# In[ ]:


from seaborn import countplot, axes_style
from matplotlib.pyplot import show,figure
from pandas import DataFrame
from IPython.display import display

with axes_style("whitegrid"):
    for feature in categorical_features + count_features:
        
        figure(figsize=(12.5,7))
        ax = countplot(data[feature], color="dimgrey")
        ax.set_title(feature)
        ax.set_xlabel("", visible=False)
        if data[feature].unique().size > 5: ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        show()
        
        display(DataFrame(data[feature].value_counts().apply(lambda x: x / len(data) * 100).round(2)).T)


# ---

# In[ ]:


from seaborn import distplot, boxplot, despine
from matplotlib.pyplot import subplot
from IPython.display import display
from pandas import DataFrame

def numeric_analysis(series):
    
    no_nulls = series.dropna()
    
    with axes_style({"axes.grid": False}):
        
        cell_1 = subplot(211)
        dp = distplot(no_nulls, kde=False)
        dp.set_xlabel("",visible=False)
        dp.set_yticklabels(dp.get_yticklabels(),visible=False)
        despine(left = True)

        cell_2 = subplot(212, sharex=cell_1)
        boxplot(no_nulls)
        despine(left=True)
    
    show()
    
    display(DataFrame(series.describe().round(2)).T)


# In[ ]:


for n in numerical_features: 
    numeric_analysis(data[data[n].notnull()][n])


# ---

# In[ ]:


from seaborn import lmplot
from matplotlib.pyplot import figure

for c in count_features:

    lmplot(data=data, x="long", y="lat", fit_reg=False, hue=c, size=10)
    show()

