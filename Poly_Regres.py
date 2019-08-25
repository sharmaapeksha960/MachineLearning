# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 18:15:25 2019

@author: HP
"""
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fitting Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#Fiiting Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualizing the Linear Regression results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth or Bluff indicator')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial Regression results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Truth or Bluff indicator')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial Regression results on Higher scale
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff indicator')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Predicting salary for level 6.5 using Linear Regression model
lin_reg.predict(np.array([[6.5]]))

#Predicting salary for level 6.5 using Polynomial regression model
lin_reg2.predict(poly_reg.fit_transform(np.array([[6.5]])))