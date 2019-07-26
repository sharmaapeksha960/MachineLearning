# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:01:05 2019

@author: HP
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values

#Handle Missing Data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#Handle Categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoder_X=LabelEncoder()
X[:,0]=labelEncoder_X.fit_transform(X[:,0])
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])],remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)

#Splitting the Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_Train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)

#Feature Scaling




