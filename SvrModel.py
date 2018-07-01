# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 22:46:00 2018

@author: priyanshumehta
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X = sc_X.fit_transform(X)

sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(-1,1))

#Fitting the Regression model to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)


#predicting results

y_pred = sc_y.inverse_transform( regressor.predict(sc_X.transform(np.array([[6.5]]))))


#Visualising SVR Model results
plt.scatter(X,y, color = 'red')
plt.plot(X, regressor.predict(X) , color= 'blue' )
plt.title('Truth Or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

