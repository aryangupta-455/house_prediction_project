import streamlit as st
import pandas as pd
import pickle
from huggingface_hub import InferenceClient

# -------------------------------
# Load trained ML model
# -------------------------------
model = pickle.load(open("house_price_model.pkl", "rb"))

# -------------------------------
# Hugging Face Inference Client
# -------------------------------
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token=st.secrets["HF_TOKEN"])

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🏠 House Price Prediction & AI Assistant")

# Sidebar inputs
st.sidebar.header("House Features")
rooms = st.sidebar.number_input("Number of Rooms", 1, 10, 3)
distance = st.sidebar.number_input("Distance from CBD (km)", 0.0, 50.0, 10.0)
bathrooms = st.sidebar.number_input("Number of Bathrooms", 1, 5, 2)
car = st.sidebar.number_input("Car Spots", 0, 5, 1)
landsize = st.sidebar.number_input("Land Size (m²)", 0.0, 1000.0, 100.0)
buildingarea = st.sidebar.number_input("Building Area (m²)", 0.0, 500.0, 100.0)
region = st.sidebar.selectbox("Region", ["Northern Metropolitan", "Southern Metropolitan", "Eastern Metropolitan", "Western Metropolitan"])

# Predict button
if st.sidebar.button("Predict Price"):
    input_data = pd.DataFrame(
        [[rooms, distance, bathrooms, car, landsize, buildingarea, region]],
        columns=["Rooms", "Distance", "Bathroom", "Car", "Landsize", "BuildingArea", "Regionname"]
    )
    price = model.predict(input_data)[0]
    st.success(f"Predicted House Price: ${price:,.0f}")

# -------------------------------
# AI Assistant
# -------------------------------
st.markdown("---")
st.markdown("### 🤖 AI Assistant: Housing & Investment Advice")
st.write("Ask me about housing, investment, or suburbs in Melbourne:")

user_input = st.text_input("Your Question:")

if user_input:
    with st.spinner("Thinking..."):
        try:
            response = client.text_generation(
                user_input,
                max_new_tokens=250,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.2
            )
            st.success(response)
        except Exception as e:
            st.error(f"⚠️ AI Error: {e}")
