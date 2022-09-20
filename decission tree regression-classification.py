#!/usr/bin/env python
# coding: utf-8

# In[111]:


import pandas as pd


# In[112]:


df = pd.read_csv("C:\\Users\\KUNAL SINGH\\OneDrive\\Desktop\\YOMA BYLD TECHNOLOGIES\\SUPERVISED LEARNING\\REGRESSION ALGORITHM\\decission tree data.csv")


# In[113]:


df.head()


# In[114]:


inputs = df.drop('salary_more_than_100k',axis = 'columns') #we have to drop the target in this algorithm 


# In[115]:


inputs


# In[116]:


target  = df['salary_more_than_100k']
target 


# In[117]:


from sklearn.preprocessing import LabelEncoder #machine leanrning is work on numeric data so have to convert the caterogical data into numeric form 
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()


# In[118]:


inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree']) #they create our data into new column in numeric form 
inputs


# In[119]:


inputs_n = inputs.drop(['company','job','degree'],axis='columns')
inputs_n


# In[120]:


from sklearn import tree
model = tree.DecisionTreeClassifier()


# In[121]:


model.fit(inputs_n,target)


# In[122]:


model.score(inputs_n,target)


# In[123]:


model.predict([[0,1,0]])


# In[124]:


#  2nd problem regression in titaninc passenger data 


# In[125]:


import pandas as pd


# In[126]:


df1= pd.read_csv("C:\\Users\\KUNAL SINGH\\OneDrive\\Desktop\\YOMA BYLD TECHNOLOGIES\\SUPERVISED LEARNING\\REGRESSION ALGORITHM\\titanic pessenger data.csv")


# In[153]:


df1


# In[154]:


#fare,age sex,pclass


# In[155]:


inputs1 = df1.drop(["PassengerId","Survived","Name","SibSp","Parch","Ticket","Cabin","Embarked"], axis = 'columns')
inputs1


# In[157]:


target1 = df1['Survived']
target1


# In[130]:


from sklearn.preprocessing import LabelEncoder


# In[131]:


le_Pclass = LabelEncoder()
le_Sex = LabelEncoder()
le_Age = LabelEncoder()
le_Fare = LabelEncoder()


# In[132]:


inputs1['Pclass_n'] = le_Pclass.fit_transform(inputs1['Pclass'])
inputs1['Sex_n'] = le_Sex.fit_transform(inputs1['Sex'])
inputs1['Age_n'] = le_Age.fit_transform(inputs1['Age'])
inputs1['Fare_n']= le_Fare.fit_transform(inputs1['Fare'])


# In[133]:


inputs1


# In[143]:


inputs1_n = inputs1.drop(['Pclass','Sex','Age','Fare'],axis ='columns')
inputs1_n


# In[141]:


#decision tree regression 
from sklearn import tree
model1 = tree.DecisionTreeClassifier()


# In[158]:


model1.fit(inputs1_n,target1)


# In[162]:


model1.score(inputs1_n,target1)


# In[164]:


model1.predict([[2,1,42,30]])


# In[ ]:




