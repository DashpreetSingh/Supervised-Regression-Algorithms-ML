# -*- coding: utf-8 -*-
"""ridge & lesso regression

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rez_ctLKgE8GHMuX9EHNYSf1FLBGNvjg
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
path = ("/content/drive/MyDrive/supervised learning project data /bangalore house price prediction OHE-data.csv")

df = pd.read_csv(path)

df.head()

"""split data 

"""

x = df.drop(['price'],axis = 'columns')
x

y = df['price']
y

x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2)

x_train.shape

x_test.shape

y_train.shape

y_test.shape

"""feature scaling


"""

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

"""linear regression model"""

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

model.score(x_test,y_test)

model.coef_

model.intercept_

"""predict the value of homes"""

x_test

x_test[0,:]

model.predict([x_test[0,:]])

model.predict(x_test)

y_test

model.score(x_test,y_test)

"""implementation of ridge and lesso """

from sklearn.linear_model import Ridge, Lasso

rd = Ridge()
rd.fit(x_train,y_train)
rd.score(x_test,y_test)

ls =  Lasso()
ls.fit(x_train,y_train)
ls.score(x_test,y_test)

