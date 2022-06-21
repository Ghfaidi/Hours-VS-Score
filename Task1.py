#!/usr/bin/env python
# coding: utf-8

# By Ghofrane Faidi

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[3]:


# Read the CSV file into a DataFrame: df
df = pd.read_csv("C:\\Users\\Ghofrane\\Desktop\\tsf.txt")


# In[4]:


df.head()


# In[5]:


# Create arrays for features and target variable
y = df['Scores'].values
X =df['Hours'].values
y_reshaped =y.reshape(-1,1)
X_reshaped =X.reshape(-1,1)


# In[6]:


plt.scatter(X_reshaped,y_reshaped)
plt.ylabel('Scores')
plt.xlabel('Hours')
plt.show;


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state=0)
X_train=X_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)


# In[31]:


#Create the regressor and fit it to the train data
reg=LinearRegression()
reg.fit(X_train,y_train)
# Predict on the test data: y_pred
y_pred =reg.predict(X_test)


# In[32]:


print("R^2: {}".format(reg.score(X_test,y_test)))


# In[39]:


#plotting the regression line
plt.scatter(X_reshaped,y_reshaped)
plt.ylabel('Scores')
plt.xlabel('Hours')
plt.title('Hours VS Score')
prediction_space = np.linspace(min(X), max(X)).reshape(-1,1)
y_pred =reg.predict(prediction_space)
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()


# In[33]:


X_train.shape


# In[34]:


q=np.array([9.25])
q.shape


# In[35]:


q=q.reshape(-1,1)


# In[36]:


res=reg.predict(q)


# In[37]:


res


# If a student studies for 9.25 hrs/day, the predicted score is 93.69173249
