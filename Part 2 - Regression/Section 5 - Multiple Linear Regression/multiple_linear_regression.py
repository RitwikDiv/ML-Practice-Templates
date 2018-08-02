# Multiple Linear Regression

# Data Preprocessing Template

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#sklearn is really good for machine models and can calc mean 

from sklearn.preprocessing import Imputer

#importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values #Independent variables, -1 indicated thatwe remove the last coulmn
y = dataset.iloc[:, 4].values #Dependent variable "Purchased"

"""
# How to deal with missing data from data set
#We can delete it or take the mean of the columns

imputer = Imputer(missing_values = 'NaN', strategy = "mean", axis = 0 )
imputer = imputer.fit(x[:, 1:3])#Upper bound is exclued so 1:3 means 1 and 2 since 1 and 2 coumns have missing data
x[:, 1:3] = imputer.transform(x[:, 1:3]) # Transform adds the values of missing data and transforms it

# How to encode categorical data since Ml models are based on mathematics, we need to change text to numbers
#Label encoding the country
"""
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
"""
#We need to dummy enocde it with each column for each country
"""
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [3]) #o is the column index here 
x = onehotencoder.fit_transform(x).toarray()

#Avoiding the dummy variable trap 
# To avoid the dummy variable problem we need to avoid one column.  
x = x[:, 1:]

"""
#Label encoding the dependent variable 

labelencoder_y = LabelEncoder()
y[:,] = labelencoder_y.fit_transform(y[:,])
"""
#Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, 
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

#We need to omit one dummy variable regardless hwo many we have


#Fitting the multiple linear regression model on the training set
from sklearn.linear_model import LinearRegression

MultipleRegressor = LinearRegression()
MultipleRegressor.fit(x_train, y_train)

#Predicting the test set results

y_pred = MultipleRegressor.predict(x_test)

#Building an optimal model using Backwards Elimination
import statsmodels.formula.api as sm
#We need to add another column which has values of one because the constant b0 in the equation has a coefficient of x0 which is one
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)
x_opt = x[:, [0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

#Implementing backward elimination
x_opt = x[:, [0,1,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,3,5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size = 0.2, 
                                                    random_state = 0)
MultipleRegressor.fit(x_train, y_train)
y_pred = MultipleRegressor.predict(x_test)