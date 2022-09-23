#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
df = pd.read_csv("C:\\Users\\KUNAL SINGH\\OneDrive\\Desktop\\YOMA BYLD TECHNOLOGIES\\SUPERVISED LEARNING\\REGRESSION ALGORITHM\data\\homepricesone.csv")


# In[2]:


df


# In[3]:


dummies = pd.get_dummies(df)


# In[4]:


dummies


# In[18]:


target = dummies.drop(["town_west windsor"],axis = "columns")
target


# In[35]:


x = target.drop("price", axis = "columns")
x


# In[36]:


y = target.price
y


# In[37]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[38]:


model.fit(x,y)


# In[40]:


model.score(x,y)


# In[43]:


model.predict([[3100,0,1]])


# In[ ]:




