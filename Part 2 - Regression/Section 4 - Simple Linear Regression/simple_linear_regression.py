#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 19:12:47 2017

@author: ANAND
"""

# Data Preprocessing Template

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#sklearn is really good for machine models and can calc mean 

from sklearn.preprocessing import Imputer

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values #Independent variables, -1 indicated thatwe remove the last coulmn
y = dataset.iloc[:, 1].values #Dependent variable "Purchased"

"""
# How to deal with missing data from data set
#We can delete it or take the mean of the columns

imputer = Imputer(missing_values = 'NaN', strategy = "mean", axis = 0 )
imputer = imputer.fit(x[:, 1:3])#Upper bound is exclued so 1:3 means 1 and 2 since 1 and 2 coumns have missing data
x[:, 1:3] = imputer.transform(x[:, 1:3]) # Transform adds the values of missing data and transforms it

# How to encode categorical data since Ml models are based on mathematics, we need to change text to numbers
#Label encoding the country

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

#We need to dummy enocde it with each column for each country

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0]) o is the column index here 
x = onehotencoder.fit_transform(x).toarray()

#Label encoding the dependent variable 

labelencoder_y = LabelEncoder()
y[:,] = labelencoder_y.fit_transform(y[:,])
"""
#Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, 
                                                    random_state = 0)
"""
#Feature scaling is important because ML uses euclidian distances
#We can scale data by standardization: xstand = x - mean(x)/standard deviation(x)
#We can scale data by Normalization :  xnorm = x - min(x)/ max(x)- min(x)
#Algroithms scale must faster if we do this

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)


#Scaling dummy variables depends on context
#In case of country they might be on same scale but you lose inerpretation
"""


# Fitting Simple Linear Regression to the training set

from sklearn.linear_model import LinearRegression
linearRegressor = LinearRegression()
linearRegressor.fit(x_train, y_train) #Teaching it to learn from x_train and y_train

# Predicting the test set results

y_pred = linearRegressor.predict(x_test)

#Visualizing the training set results

plt.scatter(x_train,y_train, color = 'red')
plt.plot(x_train, linearRegressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


#Visualizing the test set results

plt.scatter(x_test,y_test, color = 'red')
plt.plot(x_train, linearRegressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

