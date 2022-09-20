#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[4]:


df = pd.read_csv("C:\\Users\\KUNAL SINGH\\OneDrive\\Desktop\\YOMA BYLD TECHNOLOGIES\\SUPERVISED LEARNING\\REGRESSION ALGORITHM\\datasetc.csv")


# In[5]:


df


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("area(sqft)")
plt.ylabel("price(us$)")
plt.scatter(df.area,df.price)


# In[7]:


# linear regrssion

reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)


# In[9]:


reg.predict([[3300]])


# In[10]:


reg.coef_


# In[11]:


reg.intercept_


# In[12]:


135.78767123*3300+180616.43835616432


# In[ ]:




