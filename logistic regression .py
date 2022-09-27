#!/usr/bin/env python
# coding: utf-8

# # predection would a person buy a life insurance or not 

# In[2]:


import pandas as pd
import  matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


df = pd.read_csv(r"C:\Users\KUNAL SINGH\OneDrive\Desktop\YOMA BYLD TECHNOLOGIES\SUPERVISED LEARNING\REGRESSION ALGORITHM\data\logistic_regression_data.csv")


# In[59]:


df


# In[19]:


plt.scatter(df.age, df.bought_insurance, marker= "+", color = "red")


# In[23]:


from sklearn.model_selection  import train_test_split


# In[29]:


x_train , x_test , y_train, y_test = train_test_split(df[["age"]],df.bought_insurance,test_size = 0.1)


# In[38]:


x_train


# In[39]:


x_test


# In[43]:


from sklearn.linear_model import LogisticRegression


# In[51]:


model = LogisticRegression()


# In[52]:


model.fit(x_train,y_train)


# In[63]:


model.predict([[40]])


# In[65]:


model.predict_proba(x_test)


# In[66]:


model.score(x_train,y_train)


# In[67]:


model.coef_


# In[68]:


model.intercept_


# In[ ]:




