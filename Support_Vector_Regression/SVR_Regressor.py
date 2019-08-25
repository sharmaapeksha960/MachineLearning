# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 18:36:57 2019

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

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_Y=StandardScaler()
X=sc_X.fit_transform(X)
y=np.squeeze( sc_Y.fit_transform(y.reshape(-1,1)))

#Fitting SVR into dataset
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')

#Fitting the Regressor
regressor.fit(X,y)

#Visualizing the SVR Regressor
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or Bluff indicator')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Predicting the salary
y_pred=sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
#Performing inverse transform as well to get the accurate value on the above line


#Visualize on higher resolution
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff indicator')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
