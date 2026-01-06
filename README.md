# 🏡 House Price Prediction with Streamlit + LLM Enhancement

This project is an end-to-end **Machine Learning web application** that predicts **house prices** based on user input features. It is built using a **Random Forest Regressor** trained on the **Melbourne Housing Dataset**, enhanced with **LLM integration** from **Hugging Face** to make the app more interactive and educational.

> 🔗 **Live App**: [House Price Predictor](https://housepredictionproject-c3duhnpbcalvr2f3t7xaz2.streamlit.app/)  
> 💻 **GitHub Repo**: [aryangupta-455/house_prediction_project](https://github.com/aryangupta-455/house_prediction_project)

---

## 📌 Overview

House prices are influenced by many complex factors, and predicting them accurately is a valuable real-world application of machine learning. This project performs:

- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and evaluation
- Real-time prediction via a web interface
- LLM-powered explanation of predictions

---

## 🧠 Key Features

- ✅ Streamlit web app for real-time predictions  
- ✅ Random Forest Regression with 90%+ R² score  
- ✅ Preprocessing: missing values, encoding, scaling  
- ✅ Hugging Face LLM integration for user Q&A and explanation  
- ✅ Clean, intuitive UI with customizable inputs  
- ✅ Modular Python code and reusability  

---

## 📁 Dataset Used

- **Source**: Melbourne Housing Dataset  
- **Features**: Rooms, Bathroom, Landsize, BuildingArea, YearBuilt, Location, etc.  
- **Target**: Price  

### Preprocessing Includes:

- Handling missing values  
- Converting categorical variables  
- Normalizing numerical features  
- Removing outliers  

---

## 🧮 Model Details

- **Model Used**: RandomForestRegressor (Scikit-learn)  
- **Why Random Forest?** Robust to outliers, handles non-linearity, and requires minimal tuning  
- **Metric**: R² Score (>90%)  

### Evaluation Summary:

| Metric      | Value        |
|-------------|--------------|
| R² Score    | 0.91+        |
| MAE         | ~55,000 AUD  |
| RMSE        | ~78,000 AUD  |

---

🤖 LLM Integration (Google Gemini)

Integrated Google’s Gemini LLM to:

Answer user questions in natural language

Explain predictions and insights

Provide interactive, context-aware advice for housing and investment decisions

Example queries:

"Should I buy this house at this price?"

"What factors make this property a good investment?"

"Which features affect the predicted price the most?"

---

## 🏗️ Application Architecture

User Input
│
▼
Streamlit UI (app.py)
│
├──> Preprocessing (utils.py)
│
├──> Random Forest Model (model.pkl)
│
├──> LLM Response via HuggingFace API
│
▼
Prediction Output + LLM Explanation

---

## 🛠️ Tech Stack

| Category       | Tools Used                                  |
|----------------|----------------------------------------------|
| Programming    | Python                                       |
| ML Libraries   | Pandas, NumPy, Scikit-learn, Matplotlib      |
| Web Framework  | Streamlit                                    |
| LLM Integration| Hugging Face Transformers                    |
| Deployment     | Streamlit Cloud                              |
