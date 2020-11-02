#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
df = pd.read_csv('data/cancer_data.csv')
pd.options.display.max_columns = 50


# In[2]:


df.shape


# Checking data for missing values and datatypes

# In[3]:


df.dtypes


# no missing values

# In[4]:


df.describe()


# In[5]:


df.head()


# ## Data Preperation

# get rid of spaces in column headings and remove unnamed col 

# In[6]:


df.columns = [col_name.replace(' ','_') for col_name in df.columns]


# In[7]:


df.head(1)


# In[8]:


#df = df.drop(columns=['Unnamed:_0'], axis=1)
del df['Unnamed:_0']
df.head()


# ## EDA

# - See if there are relations between variables to create additional variables
#   - catagories of variables and see if the distributions clump in specific ways
#      - If so bin them
#   - look at scatterplots of cont variables
#      - if there are groupings in those scatter plots, bin the categorical variables
#      
#      - if there are trends, we can create additional variables specify the relationships
#        - if we see an exponential curve, we can create a new cariable taking the log, etc
#        
#      

# ## Modeling

# - Split test / train sets
# 
# - pick classification models to run
# 
#     - knn
#     - logistic regression
# - run initial models for each of the above to get a baseline 
#     
#     - fit X_train to model
#     - generate preditions on train set
#     - generate metrics for proictions against actual values
#         - look at recall, because we want to select the models that do the best minimizing false negatives
#         

# ### Run initial baseline models

# ### log reg

# The below is generating a recall score using the entire test set
# 
# We want to use recale to mininize false negatives, because the cost of predicting a benigng when a tumor is malignant is higher than vice versa

# In[9]:


X = df.drop('malignant', axis = 1)
y = df['malignant']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=.3, 
    random_state=1
)


# In[10]:


log_reg_vanilla = LogisticRegression(random_state=1, max_iter=10000)


# In[14]:


log_reg_vanilla.fit(X_train, y_train)
log_reg_vanilla_train_preds = log_reg_vanilla.predict(X_train)

log_reg_vanilla_train_recall = recall_score(y_train, log_reg_vanilla_train_preds)


log_reg_vanilla_train_recall


# We don't know if the above score is genrizable to other sample data
# 
# In order to generate scores we think are generizable we'll use cross- validation to generate the recall score

# ### cross validation

# In[19]:


log_reg_vanilla_cv = LogisticRegression(random_state=1, max_iter=10000)
log_reg_vanilla_crossval_train_mean = np.mean(
    cross_val_score(
        log_reg_vanilla_cv,
        X_train,
        y_train,
        scoring='recall'
    )
)

log_reg_vanilla_crossval_train_mean


# we don't have training metrics to be able to diagnose over/ underfitting we'll use cross_validate to do so

# ## Final analysis interpretation 

# In[26]:


log_reg_vanilla_ttmetrics = LogisticRegression(random_state=1, max_iter=10000)

cv_dict = cross_validate(
    log_reg_vanilla_ttmetrics,
    X_train, 
    y_train, 
    scoring='recall',
    return_train_score = True
)


# In[27]:


np.mean(cv_dict['train_score'])


# In[ ]:


np

