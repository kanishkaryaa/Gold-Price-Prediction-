#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[9]:


# loading the csv data to a pandas dataframe
gold_data = pd.read_csv('gold_price_data.csv')


# In[7]:


gold_data.head()


# In[10]:


gold_data.tail()


# In[12]:


# number of rows and columns
gold_data.shape


# In[13]:


# getting the information about the data
gold_data.info()


# In[14]:


gold_data.isnull().sum()


# In[15]:


# getting the statistical information about the data
gold_data.describe()


# In[20]:


# finding the correlation between the data columns in the dataset

# there are two types of correlation : positive and negative correlation

correlation = gold_data


# In[23]:


print(correlation['GLD'])


# In[27]:


# checking the distribution of GLD price
sns.histplot(gold_data['GLD'], color='green')


# In[28]:


# splitting the features and the target
X = gold_data.drop(['Date', 'GLD'],axis=1)
Y = gold_data['GLD']


# In[29]:


print(Y)


# In[31]:


# splitting into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[32]:


# training the model using random forest 
regressor = RandomForestRegressor(n_estimators=100)


# In[34]:


regressor.fit(X_train, Y_train)


# In[35]:


# prediction on test data
test_data_prediction = regressor.predict(X_test)


# In[36]:


print(test_data_prediction)


# In[38]:


# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)


# In[39]:


# compare the actual values and predicted values
Y_test = list(Y_test)
plt.plot(Y_test, color='blue', label='Actual Value')
plt.plot(test_data_prediction, color='green',label="predicted value")
plt.title("Actual Price vs Predicted Price")
plt.xlabel('Number of values')
plt.ylabel("GLD Price")
plt.legend()
plt.show()


# In[ ]:




