# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:26:21 2023

@author: Jayesh
"""

import pickle
from sklearn.preprocessing import PolynomialFeatures 

# Load the saved model
loaded_model= pickle.load(open('C:/Ayush/model1.sav','rb'))
# Paste the training code

poly = PolynomialFeatures(degree=2)
years=10
print(f'Prediction -house sale  Of World {2014.5+years} will be:',end=' ')
print(loaded_model.predict(poly.fit_transform([[2014+years]])))