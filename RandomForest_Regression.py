# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:01:50 2022

@author: sayan
"""

#importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values


#Training the Random Forest Regression on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 10 , random_state=0)

regressor.fit(X,y)


#Predicting the new result
regressor.predict([[6.5]])


#plot
X_grid = np.arange(min(X),max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color = 'red')
plt.plot(X_grid, regressor.predict(X_grid),color= 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()