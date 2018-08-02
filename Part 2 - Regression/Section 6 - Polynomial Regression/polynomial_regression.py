# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#sklearn is really good for machine models and can calc mean 
from sklearn.preprocessing import Imputer

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values #Independent variables, -1 indicated thatwe remove the last coulmn
y = dataset.iloc[:, 2].values #Dependent variable "Purchased"


# How to deal with missing data from data set
#We can delete it or take the mean of the columns
"""
imputer = Imputer(missing_values = 'NaN', strategy = "mean", axis = 0 )
imputer = imputer.fit(x[:, 1:3])#Upper bound is exclued so 1:3 means 1 and 2 since 1 and 2 coumns have missing data
x[:, 1:3] = imputer.transform(x[:, 1:3]) # Transform adds the values of missing data and transforms it
"""
# How to encode categorical data since Ml models are based on mathematics, we need to change text to numbers
#Label encoding the country
"""
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
"""
#We need to dummy enocde it with each column for each country
"""
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
"""
#Label encoding the dependent variable 
"""
labelencoder_y = LabelEncoder()
y[:,] = labelencoder_y.fit_transform(y[:,])
"""
#If the dataset is really small we dont need to train and we need to make a really accuate prediction

#Splitting the dataset into training set and test set
"""f
rom sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, 
                                                   random_state = 0)
"""

#Feature scaling is important because ML uses euclidian distances
#We can scale data by standardization: xstand = x - mean(x)/standard deviation(x)
#We can scale data by Normalization :  xnorm = x - min(x)/ max(x)- min(x)
#Algroithms scale must faster if we do this
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""
#Scaling dummy variables depends on context
#In case of country they might be on same scale but you lose inerpretation

#Linear Regression to compare

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)


# Polynomial Regression
'''
#visualizing the trend to find a fit
plt.scatter(x[:,], y, color = 'red')
plt.show()
'''
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
#This adds a coulmn of ones to be coeff of b0

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#Visualizing the Linear Regression Results
plt.scatter(x, y, color = "red")
plt.plot(x, lin_reg.predict(x), color = "blue")
plt.title("Linear Regression Result")
plt.xlabel("Position Levels")
plt.ylabel("Salary")
plt.show()

#Visualizing the Polynomial Linear Regression Results
#how to add additional elements to make the graph smoother
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))


plt.scatter(x, y, color = "red")
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color = "blue")
plt.title("Polynomial Linear Regression Result")
plt.xlabel("Position Levels")
plt.ylabel("Salary")
plt.show()


#Predicting a new result with linear regression

print("According to linear regression this is the salary: ",lin_reg.predict(6.5))

# Predicting a new result with polynomial regression
print("According to polynomial regression: ",
      lin_reg2.predict(poly_reg.fit_transform(6.5)))

