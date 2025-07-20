# Breast Cancer Prediction Web App

A user-friendly, interactive Streamlit web application for predicting breast cancer using a trained machine learning model.  
## 🚀 Live 

visit-https://breast-cancer-prediction21.streamlit.app

---

## 📑 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Usage Instructions](#usage-instructions)
- [Model Details](#model-details)
- [Screenshots](#screenshots)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)

---

## 📝 Project Overview

This web application allows users—regardless of technical background—to:
- Input medical parameters relevant to breast cancer.
- Receive instant and reliable malignancy predictions.
- Understand prediction logic through intuitive visualizations.

The app is intended as a proof of concept for accessible, explainable AI in healthcare.

---

## ✨ Features

- **Interactive UI:** Enter or adjust input parameters easily with sliders and input fields.
- **Real-Time Prediction:** Instantly get a benign/malignant classification based on your input.
- **Explainability:** View model feature importances for transparent predictions.
- **Visualizations:** See graphical representations (charts, plots) for model outputs.
- **Accessibility:** No installation needed—use directly in your web browser.
- **Open Source:** Full code, model, and supporting files included for learning and extension.

---

## ⚙️ How It Works

1. **Input Data:** Enter or adjust tumor/diagnostic features via the sidebar.
2. **Compute Prediction:** The app loads a pre-trained ML model to classify input data.
3. **Visual Output:** See prediction results, probability scores, and feature importance plots for interpretability.

---

## ▶️ Usage Instructions

1. **Open the App:**  
   [https://breast-cancer-prediction21.streamlit.app/](https://breast-cancer-prediction21.streamlit.app/)
2. **Input Parameters:**  
   Use the controls to provide or modify tumor-related features.
3. **Predict:**  
   Click the **Predict** button and view classification, confidence, and explanatory visuals.

---

## 🤖 Model Details

- **Model:** Random Forest Classifier (or specify other type if used)
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) – scikit-learn
- **Features:** Tumor/cell nucleus measurements (mean radius, texture, etc.)
- **Outputs:**  
  - **Diagnosis:** Benign or Malignant  
  - **Probability:** Class confidence  
  - **Feature Importances:** Visual explanation

---


## 📦 Requirements

To run locally, make sure you have:
streamlit
scikit-learn
pandas
matplotlib
joblib


---

##Project Structure

├── app.py                   # Streamlit web application
├── train_model.py           # Script to train/export the ML model
├── breast_cancer_model.pkl  # Saved machine learning model file
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation


