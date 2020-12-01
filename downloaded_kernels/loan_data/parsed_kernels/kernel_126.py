
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_csv("../input/loan.csv", low_memory=False)
df.head()


# In[ ]:


df_headers = list(df.columns.values)
df_headers


# In[ ]:


print (df.loan_status.unique())


# In[ ]:


print (df.loan_status.value_counts())


# In[ ]:


df['Default_Binary'] = int(0)
print (df.Default_Binary[0:5])


# In[ ]:


for index, value in df.loan_status.iteritems():
    if value == 'Default':
        df.set_value(index,'Default_Binary',int(1))
    if value == 'Charged Off':
        df.set_value(index, 'Default_Binary',int(1))
    if value == 'Late (31-120 days)':
        df.set_value(index, 'Default_Binary',int(1))    
    if value == 'Late (16-30 days)':
        df.set_value(index, 'Default_Binary',int(1))
    if value == 'Does not meet the credit policy. Status:Charged Off':
        df.set_value(index, 'Default_Binary',int(1))    


# In[ ]:


print (df.Default_Binary.dtype)


# In[ ]:


# make sure our default binary matches the values in the loan status column
print (df.Default_Binary.value_counts())


# In[ ]:


print (df.Default_Binary[300:350])


# In[ ]:


count = 0
for index, value in df.loan_status.iteritems():
    if count < 100:
        if value == 'Default':
            count += 1
            print ("Index of Default",index)
    else:
        print ("Done iterating")
        break


# <h1> Feature Selection and Processing </h1>

# In[ ]:


#print (df.int_rate.unique())
print (df.int_rate.value_counts())


# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat

#tf.logging.set_verbosity(tf.logging.INFO) # uncomment later and fix all the warnings
tf.logging.set_verbosity(tf.logging.FATAL) # Other options DEBUG, INFO, WARN, ERROR, FATAL


# In[ ]:


COLUMNS = ['int_rate','Default_Binary']          
FEATURES = ['int_rate']
LABEL = 'Default_Binary'


# In[ ]:


#Load datasets
print (len(df.Default_Binary))
training_set = df[0:500000] # Train on first 500k rows
testing_set = df[500001:] # Test on final ~380K rows


# In[ ]:


def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES} 
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols, labels


# In[ ]:


# Feature cols
feature_cols = [tf.contrib.layers.real_valued_column(k)
              for k in FEATURES]

# To keep only one checkpoint
#config = tf.contrib.learn.RunConfig(keep_checkpoint_max=1) ######## DO NOT DELETE

# Build 2 layer fully connected DNN with 10, 10 units respectively.
regressor = tf.contrib.learn.DNNRegressor(
  feature_columns=feature_cols, hidden_units=[10, 20, 10], ) ### REMEMBER TO ADD config=config back into arguments
  
# fit the model
regressor.fit(input_fn=lambda: input_fn(training_set), steps=751) # Boost the steps when this model starts working all the time


# In[ ]:


# Score accuracy
ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=10)
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))


# In[ ]:


prediction_set = df[850000:]
import itertools


# In[ ]:


# Print out predictions
y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
# .predict() returns an iterator; convert to a list and print predictions
predictions = list(itertools.islice(y, 37379))
#print("Predictions: {}".format(str(predictions)))

