# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#sklearn is really good for machine models and can calc mean 
from sklearn.preprocessing import Imputer

#importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, 2:4].values #Independent variables, -1 indicated thatwe remove the last coulmn
y = dataset.iloc[:, 4].values #Dependent variable "Purchased"
""""
# How to deal with missing data from data set
#We can delete it or take the mean of the columns
imputer = Imputer(missing_values = 'NaN', strategy = "mean", axis = 0 )
imputer = imputer.fit(x[:, 2:3])#Upper bound is exclued so 1:3 means 1 and 2 since 1 and 2 coumns have missing data
x[:, 1:3] = imputer.transform(x[:, 1:3]) # Transform adds the values of missing data and transforms it

# How to encode categorical data since Ml models are based on mathematics, we need to change text to numbers
#Label encoding the country

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

#We need to dummy enocde it with each column for each country
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

#Label encoding the dependent variable 
labelencoder_y = LabelEncoder()
y[:,] = labelencoder_y.fit_transform(y[:,])
"""
#Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, 
                                                    random_state = 0)
#Feature scaling is important because ML uses euclidian distances
#We can scale data by standardization: xstand = x - mean(x)/standard deviation(x)
#We can scale data by Normalization :  xnorm = x - min(x)/ max(x)- min(x)
#Algroithms scale must faster if we do this
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Scaling dummy variables depends on context
#In case of country they might be on same scale but you lose inerpretation


#Fitting logestic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

#Predicting the test set results
y_pred = classifier.predict(x_test)

#Making the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 

#([[65,  3],
  #     [ 8, 24]])
#From our matrix we got 65+24 correct predictions and 8+3 are incorrect predictions

#Visualizing the training set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#visualizing the test set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
