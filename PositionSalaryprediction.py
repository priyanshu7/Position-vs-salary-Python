# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 23:46:51 2018

@author: priyanshumehta
"""

# polynomial regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Not splitting into training set and test set as data is small

#Fitting the data in Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly , y)

#Visualising Linear Regression Model
plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg.predict(X) , color= 'blue' )
plt.title('Truth Or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#Visualising Polynomial Model
plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)) , color= 'blue' )
plt.title('Truth Or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly , y)

plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)) , color= 'blue' )
plt.title('Truth Or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly , y)


X_grid = np.arange(  min (X) , max (X) , 0.1) #for making plot continuous
X_grid = X_grid.reshape(len ( X_grid), 1)
plt.scatter(X,y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)) , color= 'blue' )
plt.title('Truth Or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#predicting results

#Linear Model
lin_reg.predict(6.5)

#Polynomial Model
lin_reg2.predict(poly_reg.fit_transform(6.5))






