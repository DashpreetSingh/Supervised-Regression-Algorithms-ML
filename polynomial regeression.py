#!/usr/bin/env python
# coding: utf-8

# In[5]:


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error , r2_score


# In[38]:


#importing the dataset
# x = data , y = quadratic equation
x = np.array(7* np.random.rand(100,1)-3)
x1 = x.reshape(-1,1)
y = 13* x*x +2 * x +7


# In[39]:


print("x :",x)
print("x1 :",x1)
print("y :",y)


# In[61]:


#data points
plt.scatter(x,y,s=10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("non-linear-data")


# # try to fit the data in the linear model 

# In[47]:


model = LinearRegression()
#fit the model 
model.fit(x1,y)


# In[69]:


print("cofficient of model is :",model.coef_)
print("intercept of model is :",model.intercept_)


# In[64]:


model.predict(x1)


# In[63]:


y1= model.predict(x1)


# In[59]:


#data points to get the output in graph 
plt.scatter(x,y,s=10)
plt.xlabel("x", fontsize = 18)
plt.ylabel("y",rotation=0,fontsize = 18)
plt.plot(x,y1,color = "g")


# # model evalution calculating the model in term of square error, root mean square error and r2 score

# In[65]:


mse = mean_squared_error(y,y1)

rmse = np.sqrt(mean_squared_error(y,y1))
r2= r2_score(y,y1)


# In[66]:


print("mse of linear model :",mse)
print("rmse of the linear model ",rmse)


# # performance of linear model is not satisfactory, lets try polynomial regression with degree 2 

# In[71]:


poly_features = PolynomialFeatures(degree= 2, include_bias = False)
x_poly = poly_features.fit_transform(x1)


# In[74]:


x[3]


# In[75]:


x_poly[3]


# In[76]:


lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)
print("coefficirnt of x is :",lin_reg.coef_)
print("intercept is :",lin_reg.intercept_)


# # this is desired quadratic equation 13x**2 + 2x+7
# # have to plot the quadratic equation 

# In[83]:


x_new = np.linspace(-3,4,100).reshape(100,1) # np.linspace is used to creating numeric sequence
x_new_poly = poly_features.transform(x_new)
y_new = lin_reg.predict(x_new_poly)
plt.plot(x,y,"b.") # b. is used for blue circle mark
plt.plot(x_new,y_new,"r",linewidth = 2,label = "prediction") # r is used to indicate the polynomial prediction
plt.xlabel("x",fontsize = 18)
plt.ylabel("y",rotation = 0,fontsize = 18)
plt.legend(loc = "upper left",  fontsize = 14)
plt.title("quadratic _prediction_plot_polynomial_regression")
plt.show()


# In[86]:


y_degree2 = lin_reg.predict(x_poly)
#model evaluation
mse_degree2 = mean_squared_error(y,y_degree)
r2_degree2 = r2_score(y,y_degree)
print("MSE of polyregression is :",mse_degree2)
print("r2 score of polyregression is :",r2_degree2)

