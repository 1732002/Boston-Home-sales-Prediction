# -*- coding: utf-8 -*-
"""

"""


import pickle
from sklearn.preprocessing import PolynomialFeatures 
import streamlit as st

# Load our pretrained model 

loaded_model= pickle.load(open('C:/Ayush/model1.sav','rb'))

# creating a function for prediction
# Pass year as a parameter as we want to take year as an input

def prediciton(years):
    poly = PolynomialFeatures(degree=2)
    print("Prediction of the price will be:",end=' ')
    return(loaded_model.predict(poly.fit_transform([[2014+years]])))


def main():
    
    # giving the title for the web app
    st.title('boston price prediction Web App')
    
    #getting the input variable

    years= st.number_input('Number of years after 2014')
    
    # code from prediction
    
    # empty string for storing the whole result
    result=''
    
    #creating a button
    if st.button('Final Prediction'):
        result=prediciton(years)  # All the input data should be mentioned in the same order as needed by the model
        
    st.success(result)
    
# Run the main function     
if __name__=='__main__':
    main()

        
    