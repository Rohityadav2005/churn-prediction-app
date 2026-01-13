import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

model = load_model('model.h5')

with open('Geo_Encoder_Obj.pkl','rb') as file:
    geo_encoder = pickle.load(file)

with open('Scaler_Obj.pkl','rb') as file:
    scaler = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder = pickle.load(file)

### note - "categories_[0]" is used by OHE to display the name of the encoded value here it gives name of the country 
### but in case of LabelEncoder we use "classes_" 


geography = st.selectbox('Geography',geo_encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder.classes_)
age = st.slider('Age',18,102)
tenure = st.slider('Tenure',0,10)
balance = st.number_input('Balance')
number_of_prod = st.slider('Number of Products',1,4)
credit_score = st.number_input('Credit Score')
Has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_mem = st.selectbox('Is an active member',[0,1])
estimated_slry = st.number_input('Estimated Salary')

st.title('Customer Churn Prediction')
##CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_prod],
    'HasCrCard': [Has_cr_card],
    'IsActiveMember': [is_active_mem],
    'EstimatedSalary': [estimated_slry]
})

geo_encoded = geo_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=geo_encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)


# scaling the data 

scaled_data = scaler.transform(input_data)

# predicting data

Prediction = model.predict(scaled_data)
Prediction_output = Prediction[0][0]

st.write(f'Churn Probability: {Prediction_output:.2f}')

if Prediction_output>0.5:
    st.write("The Customer is likely to Churn")
else:
    st.write("The Customer is not Likely to Churn")

