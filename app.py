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
st.title("🏠 Melbourne House Price Predictor")

# -----------------------------
# Load Model and Preprocessor
# -----------------------------
model = joblib.load("xgbboost_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# -----------------------------
# Hugging Face Inference Client
# -----------------------------
client = InferenceClient('tiiuae/falcon-rw-1b', token=st.secrets["hf_token"])

def get_llm_responses(user_input):
    prompt = f"""You are a real estate assistant helping users decide on housing options in Melbourne.
User question: {user_input}
Assistant:"""
    response = client.text_generation(prompt=prompt, max_new_tokens=100)
    return response.strip()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Enter Property Features")

def user_input():
    Rooms = st.number_input("Rooms", min_value=1, max_value=10, value=3)
    Bathroom = st.number_input("Bathroom", min_value=1, max_value=5, value=2)
    Bedroom2 = st.number_input("Bedroom2", min_value=1, max_value=10, value=3)
    Postcode = st.number_input("Postcode", min_value=3000, max_value=3999, value=3100)
    Landsize = st.number_input("Landsize (in sqft)", min_value=1, value=500)
    BuildingArea = st.number_input("Building Area", min_value=1, value=120)
    Type = st.selectbox("Type", ["h", "u", "t"])
    Method = st.selectbox("Method", ["S", "SP", "PI", "VB"])
    Regionname = st.selectbox("Region", [
        "Northern Metropolitan", 
        "Western Metropolitan", 
        "Southern Metropolitan", 
        "Eastern Metropolitan"
    ])
    
    return pd.DataFrame([{
        'Rooms': Rooms,
        'Bathroom': Bathroom,
        'Bedroom2': Bedroom2,
        'Postcode': Postcode,
        'Landsize': Landsize,
        'BuildingArea': BuildingArea,
        'Type': Type,
        'Method': Method,
        'Regionname': Regionname
    }])

input_df = user_input()

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):
    try:
        X_processed = preprocessor.transform(input_df)
        prediction = model.predict(X_processed)
        st.success(f"🏷️ Predicted House Price: **${int(prediction[0]):,}**")
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")

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
st.header('🤖 AI Assistant: Housing & Investment Advice')
user_query = st.text_input("💬 Ask a question about housing, investment, or suburbs")

if user_query:
    try:
        response = get_llm_responses(user_query)
        st.info(response)
    except Exception as e:
        st.error(f"❌ Error fetching AI response: {str(e)}")
