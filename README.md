# AI Crop Recommendation System

An AI-powered Crop Recommendation System developed using Machine Learning, Explainable AI (SHAP), and Google Gemini AI. The application recommends the most suitable crop based on soil nutrients and environmental conditions through an interactive Streamlit web application.


# Problem Statement

Choosing the right crop based on soil nutrients and climatic conditions is a major challenge for farmers. Incorrect crop selection can reduce productivity and profitability. This project predicts the most suitable crop using Machine Learning and provides AI-generated farming guidance.


# Project Objectives

- Predict the most suitable crop based on soil and weather parameters.
- Compare multiple Machine Learning algorithms.
- Select the best-performing model for crop recommendation.
- Explain predictions using SHAP (Explainable AI).
- Generate farmer-friendly recommendations using Google Gemini AI.
- Provide an interactive web interface using Streamlit.


## Features

-  Crop Recommendation using Random Forest
-  Confidence Score for Predictions
-  SHAP Explainability
-  AI Farming Guide powered by Google Gemini
-  Interactive Streamlit Interface
-  User-friendly Design


##  Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Matplotlib
- Joblib
- Google Gemini API


##  Dataset

Dataset Name: Crop Recommendation Dataset

Source: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

# Dataset Details

- Total Samples: 2200
- Input Features: 7
- Crop Classes: 22

# Input Parameters

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall


## Machine Learning Models Used

- Decision Tree
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest  (Selected Model)


## Explainable AI

The project uses SHAP (SHapley Additive Explanations) to interpret predictions by showing the contribution of each feature to the recommended crop.

---

## Google Gemini Integration

The application uses Google Gemini AI to generate simple and practical farming recommendations based on:

- Predicted Crop
- Soil Nutrient Values
- Temperature
- Humidity
- Rainfall
- Soil pH

# Project Structure

AI-Crop-Recommendation-System/
│
├── app.py
├── crop_model.pkl
├── Crop_recommendation.csv
├── requirements.txt
├── style.css
├── README.md
├── .gitignore
└── .streamlit/


##  Installation

### Clone the repository

```bash
git clone https://github.com/aditivaidwan07-byte/AI-Crop-Recommendation-System.git
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## Future Enhancements

- Model Performance Comparison Dashboard
- Prediction History
- Downloadable Reports
- Multi-language Support
- Cloud Deployment
- Mobile-Friendly Interface


##  Developer

Aditi
Harshita
Lakshita

B.Tech Mechanical Engineering

Indira Gandhi Delhi Technical University for Women (IGDTUW)
