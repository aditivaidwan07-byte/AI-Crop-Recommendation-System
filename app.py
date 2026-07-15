import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import google.generativeai as genai

# ===============================
# PAGE CONFIGURATION
# ===============================

st.set_page_config(
    page_title="AI Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)
st.markdown("""
<style>

.main {
    background-color: #f5fff6;
}

.stButton>button{
    background-color:#2E8B57;
    color:white;
    border-radius:10px;
    height:50px;
    width:100%;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background-color:#1f6f43;
    color:white;
}

div[data-testid="stMetric"]{
    background-color:#f0fff0;
    padding:10px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.image(
    "https://img.icons8.com/color/96/plant-under-sun.png",
    width=90
)

st.sidebar.title("AI Crop Recommendation")

st.sidebar.success("Random Forest Accuracy\n99.32%")

st.sidebar.markdown("---")

st.sidebar.markdown("""
###  Developed By

Aditi

Harshita

Lakshita

---

###  Technologies

 Python

 Streamlit

 Random Forest

 SHAP

 Google Gemini

---

###  Domain

Agriculture
""")

def load_css():

    with open("style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()
# ===============================
# LOAD MODEL
# ===============================

model = joblib.load("crop_model.pkl")

# ===============================
# GEMINI API
# ===============================

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)

gemini = genai.GenerativeModel("gemini-2.5-flash")

# ===============================
# TITLE
# ===============================

st.title("🌱 AI Based Crop Recommendation System")

st.markdown("""

### Smart Agriculture using Machine Learning & Generative AI

Predict the most suitable crop using soil nutrients and weather conditions.
""")
st.divider()

# ===============================
# USER INPUT
# ===============================

st.subheader("Enter Soil and Weather Details")

N = st.number_input(
    "Nitrogen (N)",
    min_value=0,
    max_value=150,
    value=90
)

P = st.number_input(
    "Phosphorus (P)",
    min_value=0,
    max_value=150,
    value=42
)

K = st.number_input(
    "Potassium (K)",
    min_value=0,
    max_value=250,
    value=43
)

temperature = st.number_input(
    "Temperature (°C)",
    value=20.8
)

humidity = st.number_input(
    "Humidity (%)",
    value=82.0
)

ph = st.number_input(
    "Soil pH",
    value=6.5
)

rainfall = st.number_input(
    "Rainfall (mm)",
    value=202.9
)

st.divider()
# ===============================
# PREDICTION
# ===============================

if st.button("🌱 Recommend Crop"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "N": [N],
        "P": [P],
        "K": [K],
        "temperature": [temperature],
        "humidity": [humidity],
        "ph": [ph],
        "rainfall": [rainfall]
    })

    # Predict crop
    prediction = model.predict(input_data)[0]

    # Predict probabilities
    probabilities = model.predict_proba(input_data)[0]

    # Get class names
    classes = model.classes_

    # Create dataframe of probabilities
    prob_df = pd.DataFrame({
        "Crop": classes,
        "Confidence": probabilities * 100
    })

    # Sort by confidence
    prob_df = prob_df.sort_values(
        by="Confidence",
        ascending=False
    ).reset_index(drop=True)

    confidence = prob_df.iloc[0]["Confidence"]

    # Display result
    st.markdown(f"""
##  Recommended Crop

# **{prediction.title()}**
""")

    st.info(f" Prediction Confidence: **{confidence:.2f}%**")

    # Top 3 recommendations
    st.subheader("🥇 Top 3 Recommended Crops")

    st.dataframe(
        prob_df.head(3),
        use_container_width=True
    )
 

    st.subheader("📊 Why was this crop recommended?")

    try:
        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(input_data)

        # Get the predicted class index
        class_index = list(classes).index(prediction)

        # SHAP values for predicted class
        values = np.abs(shap_values[0, :, class_index])

        importance = pd.DataFrame({
            "Feature": input_data.columns,
            "Importance": values
        })

        importance = importance.sort_values(
            by="Importance",
            ascending=False
        )

        st.bar_chart(
            importance.set_index("Feature")
        )

    except Exception as e:
        st.warning(f"SHAP Error: {e}")

    # ===============================
    # GEMINI AI EXPLANATION
    # ===============================

    st.subheader(" AI Farming Guide")

    prompt = f"""
    The predicted crop is {prediction}.

    Soil Information:
    Nitrogen = {N}
    Phosphorus = {P}
    Potassium = {K}
    Temperature = {temperature}
    Humidity = {humidity}
    pH = {ph}
    Rainfall = {rainfall}

    Prediction confidence = {confidence:.2f}%

    Explain in simple language:

    1. Why this crop is suitable.
    2. Best sowing season.
    3. Fertilizer recommendation.
    4. Irrigation advice.
    5. Common diseases and prevention.

    Keep the answer simple and farmer-friendly.
    """

    try:
        response = gemini.generate_content(prompt)
        st.success(response.text)

    except Exception as e:
        st.error(f"Gemini Error: {e}")
# ===============================
# FOOTER
# ===============================

st.divider()

st.markdown("""
---
### 🌱 AI Crop Recommendation System

**Machine Learning Model:** Random Forest

**Model Accuracy:** 99.32%

**Explainable AI:** SHAP

**Generative AI:** Google Gemini

Developed using **Python, Streamlit, Scikit-Learn, SHAP and Gemini AI**
""")