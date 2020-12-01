
# coding: utf-8

# **Introduction**
# In this post, you will discover the Keras Python library that provides a clean and convenient way to create a range of deep learning models on top of Theano or TensorFlow.
# 
# All creidts to -- "http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/"
# 
# Let’s get started.

# **Dependencies**
# 
# All important libraries and data set are imported below
# 
# **Python**
# 
# Please run this script in Python 2 

# In[2]:


import os, sys, re
import cPickle as pickle
from keras.models import Sequential
from keras.layers import Dense
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

print (time.time())
dataset = pd.read_csv("loan.csv")


# Replace all the missing entries with zeros

# In[ ]:


dataset = dataset.fillna(0) ## filling missing values with zeros


# **Data Modification**
# 
# Convert all kind of categorical data into integral values accordingly and 'Date Column' into real values'

# In[ ]:


dataset['application_type'] = dataset['application_type'].astype('category').cat.codes
dataset['addr_state'] = dataset['addr_state'].astype('category').cat.codes
dataset['earliest_cr_line'] = pd.to_datetime(dataset['earliest_cr_line'])
dataset['earliest_cr_line'] = (dataset['earliest_cr_line']-dataset['earliest_cr_line'].min())/np.timedelta64(1,'D')
dataset['emp_length'] = dataset['emp_length'].astype('category').cat.codes
dataset['grade'] = dataset['grade'].astype('category').cat.codes
dataset['home_ownership'] = dataset['home_ownership'].astype('category').cat.codes
dataset['initial_list_status'] = dataset['initial_list_status'].astype('category').cat.codes
dataset['issue_d'] = pd.to_datetime(dataset['issue_d'])
dataset['issue_d'] = (dataset['issue_d']-dataset['issue_d'].min())/np.timedelta64(1,'D')
dataset['last_credit_pull_d'] = pd.to_datetime(dataset['last_credit_pull_d'])
dataset['last_credit_pull_d'] = (dataset['last_credit_pull_d']-dataset['last_credit_pull_d'].min())/np.timedelta64(1,'D')
dataset['last_pymnt_d'] = pd.to_datetime(dataset['last_pymnt_d'])
dataset['last_pymnt_d'] = (dataset['last_pymnt_d']-dataset['last_pymnt_d'].min())/np.timedelta64(1,'D')
dataset['loan_status'] = dataset['loan_status'].astype('category').cat.codes
dataset['next_pymnt_d'] = pd.to_datetime(dataset['next_pymnt_d'])
dataset['next_pymnt_d'] = (dataset['next_pymnt_d']-dataset['next_pymnt_d'].min())/np.timedelta64(1,'D')
dataset['purpose'] = dataset['purpose'].astype('category').cat.codes
dataset['pymnt_plan'] = dataset['pymnt_plan'].astype('category').cat.codes
dataset['sub_grade'] = dataset['sub_grade'].astype('category').cat.codes
dataset['term'] = dataset['term'].astype('category').cat.codes
dataset['verification_status'] = dataset['verification_status'].astype('category').cat.codes
dataset['verification_status_joint'] = dataset['verification_status_joint'].astype('category').cat.codes


# Storing non numeric or non real columns name in non_numerics array

# In[ ]:


non_numerics = [x for x in dataset.columnsif not (dataset[x].dtype == np.float64 or dataset[x].dtype == np.int8 or dataset[x].dtype == np.int64)]


# Droping non_numerics column for easy modeling

# In[ ]:


df = dataset
df = df.drop(non_numerics,1)


# Converting 'loan result status' into two categories 0 and 1. 0 means loan failed or that type of person should not be given loan in future and 1 means loan passed i.e. they are good for extending the loan.

# In[ ]:


def LoanResult(status):
    if (status == 5) or (status == 1) or (status == 7):
        return 1
    else:
        return 0

df['loan_status'] = df['loan_status'].apply(LoanResult)


# Splitting data into train data and test data with the help of scikit library in the ratio of 3:1

# In[ ]:


train, test = train_test_split(df, test_size = 0.25)

##running complete data set will take a lot of time, hence reduced the data set
X_train = train.drop('loan_status',1).values[0:50000, :]
Y_train = train['loan_status'].values[0:50000]

X_test = test.drop('loan_status',1).values[0:1000, :]
Y_test = test['loan_status'].values[0:1000]

X_pred = test.drop('loan_status',1).values[1001:2000, :]


# Setting the seed for pseudo random numbers generation

# In[ ]:


seed = 8 
np.random.seed(seed)


# Now we will define a three layered neural network model. We create a Sequential model and add layers one at a time until we are happy with our network topology. After that we will set activation function and number of nets in each layer. These are done by heuristics and training the model several times.

# In[ ]:


# Create the model 
model = Sequential()

# Define the three layered model
model.add(Dense(110, input_dim = 68, kernel_initializer = "uniform", activation = "relu"))
model.add(Dense(110, kernel_initializer = "uniform", activation = "relu"))
model.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))


# Now we will compile the model. In this we have to input three parameters viz. loss function, optimizer function and an evaluation metrics. These choices are again by heuristics. Here we are using "binary_crossentropy" as loss func, "adam" as optimizer func and "accuracy" as evaluation metrics.

# In[ ]:


#
# Compile the model
model.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
#


# Now we have to fit the data into our model. 
# We can train or fit our model on our loaded data by calling the fit() function on the model.
# 
# The training process will run for a fixed number of iterations through the dataset called epochs, that we must specify using the **epochs** argument. We can also set the number of instances that are evaluated before a weight update in the network is performed, called the batch size and set using the **batch_size** argument.

# In[ ]:


# Fit the model
model.fit(X_train, Y_train, epochs= 22000, batch_size=200)


# **Evaluate Model**
# 
# We have trained our neural network on the entire dataset and we can evaluate the performance of the network on the test dataset.

# In[ ]:


performance = model.evaluate(X_test, Y_test)
print("%s: %.2f%%" % (model.metrics_names[1], performance[1]*100))
#


# **Final Prediction**
# 
# Predicting using the trained model

# In[ ]:


# Predict using the trained model
prediction = model.predict(X_pred)
rounded_predictions = [round(x) for x in prediction]
print(rounded_predictions)

