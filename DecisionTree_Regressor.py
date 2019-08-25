# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 23:07:04 2019

@author: HP
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fitting the Decision tree
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#Predicting new result
regressor.predict(np.array([[6.5]]))

#Visualize the result
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or Bluff Indicator')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualizing result in higher dimension
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff Indicator')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
