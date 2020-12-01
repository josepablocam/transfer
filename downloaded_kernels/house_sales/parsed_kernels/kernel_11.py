#!/usr/bin/env python
# coding: utf-8

# # Finding the most correlating variables for house price prediction

# This is a step-by-step tutorial describing a ways to find the most correlating variables for the data available in [House Sales in King County, USA](https://www.kaggle.com/harlfoxem/housesalesprediction) dataset.

# ## 1. Dataset overview

# Let's first look into dataset in order to better understand what kind of data is available:

# In[ ]:


import pandas as pd

# Always display all the columns
pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 60) 

dataset = pd.read_csv("../input/kc_house_data.csv") # read the dataset

dataset.head(5) # print 5 first rows from the dataset


# In[ ]:


dataset.dtypes # get an overview of data types presented in the dataset


# ### 1.1 Dataset quality

# Before analysing the dataset we have to make sure that it doesn't contain messy data and eventually fix it.

# #### 1.1.1 Looking for NaN values

# In[ ]:


print((dataset.isnull().any()))


# As you can see none of the columns have NaN values in it, so that we're safe to move forward with further analysis.

# ### 1.2 Identifying the variables

# Now it's time to identify which types of variables we have in the dataset.
# 
# We'll be trying to identify columns with continuous and categorical variables.
# 
# First of all let's take a look on the list of the columns which can be potentially categorical:
# - bedrooms
# - bathrooms
# - floors
# - waterfront
# - view
# - condition
# - grade

# In[ ]:


# let's observe unique values presented in potentially categorical columns
print("bedrooms")
print((sorted(dataset.bedrooms.unique())))
print("bathrooms")
print((sorted(dataset.bathrooms.unique())))
print("floors")
print((sorted(dataset.floors.unique())))
print("waterfront")
print((sorted(dataset.waterfront.unique())))
print("view")
print((sorted(dataset.view.unique())))
print("condition")
print((sorted(dataset.condition.unique())))
print("grade")
print((sorted(dataset.grade.unique())))


# As the one can see, we have following two subtypes of categorical variables here:
# - Dichotomous variable (having 2 possible values)
#     - watefront
# - Polytomous variables (having multiple possible values)
#     - bedrooms
#     - bathrooms
#     - floors
#     - view
#     - condition
#     - grade

# It would make sense to convert categorical variables from above which have continuous set of values available to "category" in our dataset in order to get better overview of them in the next step.
# 
# Additionally, let's remove variables which won't be participating in the analysis:

# In[ ]:


# Create new categorical variables
dataset['waterfront'] = dataset['waterfront'].astype('category',ordered=True)
dataset['view'] = dataset['view'].astype('category',ordered=True)
dataset['condition'] = dataset['condition'].astype('category',ordered=True)
dataset['grade'] = dataset['grade'].astype('category',ordered=False)

# Remove unused variables
dataset = dataset.drop(['id', 'date'],axis=1)

dataset.dtypes # re-check data types in the dataset after conversion above


# Let's assume all other columns than identified as categorical contain continuous variables.

# ## 2. Correlation between variables

# Our main reason exploring this dataset is to find variables which have a strong correlation with the house prices.
# 
# Let's start from the categorical variables which we defined in the previous steps and observe the correlation between them and house prices.

# ### 2.1 Categorical variables

# Recalling the categorical variables we identified earlier, let's calculate their correlation to price.
# 
# But before we begin with that, let's review our dataset and identify other variables which could bring us more value being categorical rather than continuous.
# 
# From the previous steps we can see that both "sqft_basement" and "yr_renovated" contain "0" values for the houses which either don't have basements or haven't been renovated yet. 
# 
# Taking into account this information we could verify our hypthosis that the fact that the house has basement or have been renovated may affect its price. 
# 
# We need to introduce two new categorical variables for this purpose:

# In[ ]:


dataset['basement_is_present'] = dataset['sqft_basement'].apply(lambda x: 1 if x > 0 else 0)
dataset['basement_is_present'] = dataset['basement_is_present'].astype('category', ordered = False)

dataset['is_renovated'] = dataset['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
dataset['is_renovated'] = dataset['is_renovated'].astype('category', ordered = False)

dataset.dtypes


# Now we're ready to start calculating a correlation between categorical variables and house prices in order to estimate which variable affect the house prices at most.
# 
# However, we want to subdivide our categorical variables into two subcategories: dichotomous and polytomous ones. This has effect on which correlation calculation methods to be applied to those sub-categories.
# 
# Taking into account two newly introduced variable we have a following sub-division of categorical variables:
# - Dichotomous variables:
#     - watefront
#     - basement_is_present
#     - is_renovated
# - Polytomous variables:
#     - bedrooms
#     - bathrooms
#     - floors
#     - view
#     - condition
#     - grade
#     
# Dichotomous or binary variables are going to get their correlation calculated by means point biserial correlation and polytomous ones will be treated with Spearman's rank-order correlation correspondingly:

# In[ ]:


from scipy import stats

CATEGORICAL_VARIABLES = ["waterfront", 
                       "basement_is_present", 
                       "is_renovated", 
                       "bedrooms", 
                       "bathrooms", 
                       "floors", 
                       "view", 
                       "condition",
                       "grade"]

for c in CATEGORICAL_VARIABLES:
    if c not in ["waterfront", "basement_is_present", "is_renovated"]:
        correlation = stats.pearsonr(dataset[c], dataset["price"])
    else:
        correlation = stats.pointbiserialr(dataset[c], dataset["price"])
    print(("Correlation of %s to price is %s" %(c, correlation)))


# As you can see top 3 categorical variables which have the highest correlation coefficients are:
# 1. grade (0.66)
# 2. bathrooms (0.52)
# 3. view (0.39)
# 
# Our assumption, however, that "basement_is_present" and "is_renovated" are strongly correlated with house prices is wrong.

# ### 2.2 Continuous variables

# Continuous variables will be treated similarly as categorical ones with one exception: we'll be using correlation heatmap in order to analyse the correlation of continuous variables in order to try how comfortable this visual approach is.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

CONTINUOUS_VARIABLES = ["price", 
                       "sqft_living", 
                       "sqft_lot", 
                       "sqft_above", 
                       "sqft_basement", 
                       "yr_built", 
                       "yr_renovated", 
                       "zipcode",
                       "lat",
                       "long",
                       "sqft_living15",
                       "sqft_lot15"]

# create new dataframe containing only continuous variables
cont_variables_dataframe = dataset[CONTINUOUS_VARIABLES]
# calculate correlation for all continuous variables
cont_variables_correlation = cont_variables_dataframe.corr()

# plot the heatmap showing calculated correlations
plt.subplots(figsize=(11, 11))
plt.title('Pearson Correlation of continous features')
ax = sns.heatmap(cont_variables_correlation, 
                 annot=True, 
                 linewidths=.5, 
                 cmap="YlGnBu",
                 square=True
                );


# As you can see the top 3 continuous variables in terms of correlating to house prices are:
# - sqft_living (0.7)
# - sqft_above (0.61)
# - sqft_living15 (0.59)

# ## 3. Conclusion

# We were able to identify categorical and continuous variables in our dataset and calculate their correlation to house prices.
# 
# As a result we got a list of 6 top performing va riables which may be used as features in linear and multivariate linear regression models for predicting house prices:
# 
#  - sqft_living (0.7) 
#  - grade (0.66)
#  - sqft_above (0.61)
#  - sqft_living15 (0.59)
#  - bathrooms (0.52)
#  - view (0.39)
