# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 21:42:17 2019

@author: HP
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

#Encoding Categorical variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoder_X=LabelEncoder()
X[:,3]=labelEncoder_X.fit_transform(X[:,3])
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])],remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building an optimal model using backward elimination
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
import statsmodels.formula.api as sm
#Optimal model for all possible parameters
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()

#x2 has maximum p value so removing the predictor
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()

#x2 has maximum p value so removing the predictor
X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()

#x2 has maximum p value so removing the predictor
X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()

#x2 has maximum p value so removing the predictor
X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()