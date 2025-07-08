import streamlit as st
import pandas as pd
import joblib
from transformers import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import os

st.title("🏠 Melbourne House Price Predictor (Local Version)")

model = joblib.load("xgbboost_model.pkl")
preprocessor = joblib.load('preprocessor.pkl')


#log transformer 
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X)


st.sidebar.header("Enter Property Features")

#llm model function

llm = pipeline('text_generation', model = 'tiiuae/falcon-7b-instruct')
def get_llm_responses(user_input):
    prompt = f'User: {user_input}\nAssistant:'
    response = llm(prompt,max_length = 200, do_sample = True)
    return response[0]['generated_text']

#user inputs or model inputs 

def user_input():
    Rooms = st.number_input("Rooms", 1, 10)
    Bathroom = st.number_input("Bathroom", 1, 5)
    Bedroom2 = st.number_input("Bedroom2", 1, 10)
    Postcode = st.number_input("Postcode", 3000, 3999)
    Landsize = st.number_input("Landsize (in sqft)", 1)
    BuildingArea = st.number_input("Building Area", 1)
    Type = st.selectbox("Type", ["h", "u", "t"])
    Method = st.selectbox("Method", ["S", "SP", "PI", "VB"])
    Regionname = st.selectbox("Region", ["Northern Metropolitan", "Western Metropolitan", "Southern Metropolitan", "Eastern Metropolitan"])

    data = {
        'Rooms': Rooms,
        'Bathroom': Bathroom,
        'Bedroom2': Bedroom2,
        'Postcode': Postcode,
        'Landsize': Landsize,
        'BuildingArea': BuildingArea,
        'Type': Type,
        'Method': Method,
        'Regionname': Regionname
    }

    return pd.DataFrame([data])

input_df = user_input()

# Predict
if st.button("Predict Price"):
    X_processed = preprocessor.transform(input_df)
    prediction = model.predict(X_processed)
    st.success(f"Predicted House Price: ${int(prediction[0]):,}")


#LLM Assistant

st.header('Get help of AI')
user_query = st.text_input("Ask anything about hosing, buying tios, regions and others")

if user_query:
    llm_response  =get_llm_responses(user_query)
    st.info(llm_response)
