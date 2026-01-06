import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import InferenceClient
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# -----------------------------
# Custom Log Transformer
# -----------------------------
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.log1p(X)

# -----------------------------
# Title
# -----------------------------
st.title(" Melbourne House Price Predictor")

# -----------------------------
# Load Model and Preprocessor
# -----------------------------
model = joblib.load("xgbboost_model.pkl")   # ✅ corrected filename
preprocessor = joblib.load("preprocessor.pkl")

# -----------------------------
# Hugging Face Inference Client
# -----------------------------
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=st.secrets["hf_token"]
)

def get_llm_responses(user_input):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful real estate assistant for Melbourne housing and investment advice."
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=150,
        temperature=0.7
    )

    return response.choices[0].message.content


# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("This model is trained on Melbourne data and includes an AI assistant for your help.")

def user_input():
    Rooms = st.number_input("Rooms", 1, 10)
    Bathroom = st.number_input("Bathroom", 1, 5)
    Bedroom2 = st.number_input("Bedroom2", 1, 10)
    Postcode = st.number_input("Postcode", 3000, 3999)
    Landsize = st.number_input("Landsize (in sqft)", 1)
    BuildingArea = st.number_input("Building Area", 1)
    Distance = st.number_input("Distance from CBD (in km)", 0.0, 50.0)
    price_per_sqrft = st.number_input("Price per square foot (optional estimate)", 100.0)

    Type = st.selectbox("Type", ["h", "u", "t"])
    Method = st.selectbox("Method", ["S", "SP", "PI", "VB"])
    Regionname = st.selectbox("Region", [
        "Northern Metropolitan", "Western Metropolitan", "Southern Metropolitan", "Eastern Metropolitan"
    ])

    data = {
        'Rooms': Rooms,
        'Bathroom': Bathroom,
        'Bedroom2': Bedroom2,
        'Postcode': Postcode,
        'Landsize': Landsize,
        'BuildingArea': BuildingArea,
        'Distance': Distance,
        'price_per_sqrft': price_per_sqrft,
        'Type': Type,
        'Method': Method,
        'Regionname': Regionname
    }

    return pd.DataFrame([data])


input_df = user_input()

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):
    try:
        X_processed = preprocessor.transform(input_df)
        prediction = model.predict(X_processed)
        st.success(f" Predicted House Price: **${int(prediction[0]):,}**")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

# -----------------------------
# Region Guide
# -----------------------------
with st.expander("📍 Region Guide"):
    st.markdown("""
- **Northern Metropolitan**: Budget-friendly, great for first-time buyers.
- **Western Metropolitan**: Good infrastructure, family-oriented.
- **Southern Metropolitan**: Premium area, near beaches.
- **Eastern Metropolitan**: Green suburbs, higher appreciation.
""")

# -----------------------------
# LLM Assistant
# -----------------------------
st.header('AI Assistant: Housing & Investment Advice')
user_query = st.text_input("Ask a question about housing, investment, or suburbs")

if user_query:
    try:
        response = get_llm_responses(user_query)
        st.info(response)
    except Exception as e:
        st.error(f"Error fetching AI response: {str(e)}")





