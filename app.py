import streamlit as st
import pandas as pd
import joblib
import os

st.title("🏠 Melbourne House Price Predictor (Local Version)")

model = joblib.load("house_price_model.pkl")

st.sidebar.header("Enter Property Features")

# Inputs matching your model's features
rooms = st.sidebar.slider("Rooms", 1, 10, 3)
bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 1)
car = st.sidebar.slider("Car Spaces", 0, 5, 1)
land_size = st.sidebar.number_input("Land Size (sqm)", 0, 10000, 200)
building_area = st.sidebar.number_input("Building Area (sqm)", 0, 1000, 120)
year_built = st.sidebar.slider("Year Built", 1800, 2025, 2000)
distance = st.sidebar.slider("Distance to CBD (km)", 0.0, 50.0, 10.0, step=0.1)
latitude = st.sidebar.number_input("Latitude", -38.0, -37.0, -37.8)
longitude = st.sidebar.number_input("Longitude", 144.0, 145.5, 144.9)
property_count = st.sidebar.number_input("Property Count in Suburb", 0, 10000, 5000)

# DataFrame for prediction
input_df = pd.DataFrame({
    "Rooms": [rooms],
    "Bedroom2": [bedrooms],
    "Bathroom": [bathrooms],
    "Car": [car],
    "Landsize": [land_size],
    "BuildingArea": [building_area],
    "YearBuilt": [year_built],
    "Distance": [distance],
    "Lattitude": [latitude],
    "Longtitude": [longitude],
    "Propertycount": [property_count]
})

# Run prediction
if model and st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")
