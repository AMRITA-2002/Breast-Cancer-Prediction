import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🏥", layout="wide")
st.title("🏥 Breast Cancer Prediction App")

# -------------- Model Loading Utilities --------------

@st.cache_resource
def load_model():
    model_dir = Path("models")
    model = joblib.load(model_dir / "breast_cancer_model.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")
    label_encoders = joblib.load(model_dir / "label_encoders.pkl")
    target_encoder = joblib.load(model_dir / "target_encoder.pkl")
    feature_columns = joblib.load(model_dir / "feature_columns.pkl")
    return model, scaler, label_encoders, target_encoder, feature_columns

def load_data():
    try:
        df = pd.read_csv("data/Breast_Cancer.csv")
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return None

model, scaler, label_encoders, target_encoder, feature_columns = load_model()
df = load_data()

# -------------- Sidebar Navigation --------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🔮 Predict",
        "📊 Data Exploration",
        "🧠 Feature Importance",
    ]
)

# -------------- Home Page --------------

if page == "🏠 Home":
    st.subheader("Welcome!")
    st.write(
        """
        This app predicts breast cancer patient outcomes using a machine learning model.
        - Use **Predict** to input patient characteristics and get a prediction.
        - Explore feature importance (**Feature Importance**) and dataset stats (**Data Exploration**) below.
        """
    )
    st.info(
        f"**🤖 Model:** RandomForestClassifier\n"
        f"**Trained on:** {len(df) if df is not None else '[not available]'} records\n"
        f"**Features:** {', '.join(feature_columns)}"
    )

# -------------- Prediction Page --------------

elif page == "🔮 Predict":
    st.header("Make a Prediction")
    with st.form("prediction_form"):
        user_inputs = {}
        col1, col2, col3 = st.columns(3)

        # For consistent column assignment, use a loop index
        for idx, col_handle in enumerate(feature_columns):

            # Target which column to use for UI
            target_col = col1 if idx % 3 == 0 else col2 if idx % 3 == 1 else col3

            with target_col:
                # If the feature is categorical and has a label encoder, offer its options
                if col_handle in label_encoders:
                    options = list(label_encoders[col_handle].classes_)
                    user_inputs[col_handle] = st.selectbox(col_handle, options)
                elif col_handle.lower() in ["grade"]:
                    user_inputs[col_handle] = st.selectbox("Grade", [1, 2, 3, 4])
                elif col_handle.lower() in ["t stage"]:
                    user_inputs[col_handle] = st.selectbox("T Stage", ["T1", "T2", "T3", "T4"])
                elif col_handle.lower() in ["n stage"]:
                    user_inputs[col_handle] = st.selectbox("N Stage", ["N1", "N2", "N3"])
                elif col_handle.lower() in ["6th stage"]:
                    user_inputs[col_handle] = st.selectbox("6th Stage", ["I", "II", "III", "IV", "V"])
                elif col_handle.lower() in ["a stage"]:
                    user_inputs[col_handle] = st.selectbox("A Stage", ["A", "B"])
                elif col_handle.lower() == "age":
                    user_inputs[col_handle] = st.number_input("Age", min_value=10, max_value=100, value=50)
                elif col_handle.lower() == "tumor size":
                    user_inputs[col_handle] = st.number_input("Tumor Size (mm)", min_value=1, max_value=150, value=30)
                elif col_handle.lower() == "regional node examined":
                    user_inputs[col_handle] = st.number_input("Regional Nodes Examined", 0, 50, 10)
                elif col_handle.lower() == "reginol node positive":
                    user_inputs[col_handle] = st.number_input("Regional Nodes Positive", 0, 25, 1)
                elif col_handle.lower() == "survival months":
                    user_inputs[col_handle] = st.number_input("Survival Months", 1, 150, 60)
                else:
                    # Default for any missing/unknown feature
                    user_inputs[col_handle] = 0

        submitted = st.form_submit_button("Predict")

        if submitted:
            # Convert inputs to DataFrame (single row)
            X_input = pd.DataFrame([user_inputs])

            # Encode categorical features
            for col, encoder in label_encoders.items():
                if col in X_input.columns:
                    try:
                        X_input[col] = encoder.transform(X_input[col].astype(str))
                    except:
                        X_input[col] = 0  # Unknown category fallback

            # Ensure all columns numeric and correct order
            for col in X_input.columns:
                if not pd.api.types.is_numeric_dtype(X_input[col]):
                    X_input[col] = pd.to_numeric(X_input[col], errors="coerce").fillna(0)
            for feat in feature_columns:
                if feat not in X_input.columns:
                    X_input[feat] = 0
            X_input = X_input[feature_columns]

            # Scale features, make prediction
            X_scaled = scaler.transform(X_input)
            pred_idx = model.predict(X_scaled)[0]
            probs = model.predict_proba(X_scaled)[0]
            pred_label = target_encoder.inverse_transform([pred_idx])[0]

            color = "green" if pred_label.lower() == "alive" else "red"
            st.markdown(f"### 🎯 Prediction: <span style='color:{color};font-weight:bold'>{pred_label}</span>", unsafe_allow_html=True)

            # Display probabilities as chart
            fig = go.Figure(
                data=[go.Bar(
                    x=list(target_encoder.classes_),
                    y=probs,
                    marker_color=["#51cf66" if c.lower() == "alive" else "#ff6b6b" for c in target_encoder.classes_],
                    text=[f"{p:.1%}" for p in probs],
                    textposition="auto"
                )]
            )
            fig.update_layout(title="Prediction Probabilities", yaxis_title="Probability", xaxis_title="Class")
            st.plotly_chart(fig, use_container_width=True)

# -------------- Data Exploration --------------

elif page == "📊 Data Exploration":
    st.header("Training Data Exploration")
    if df is None:
        st.warning("Please add your CSV to 'data/Breast_Cancer.csv' for this feature.")
    else:
        st.write("Sample data:")
        safe_df = df.copy()
        # Arrow/Streamlit compatibility
        if 'dtype' in safe_df.columns:
            safe_df = safe_df.rename(columns={'dtype': 'dtype_col'})
        for col in safe_df.select_dtypes(include='object').columns:
            safe_df[col] = safe_df[col].astype(str)
        st.dataframe(safe_df.head(10))  # show sample

        st.write("Column details:")
        meta_df = pd.DataFrame({
            "dtype": safe_df.dtypes.astype(str),
            "NaNs": safe_df.isnull().sum()
        })
        st.dataframe(meta_df)

        # Status distribution
        if "Status" in safe_df.columns:
            st.subheader("Status Distribution")
            st.bar_chart(safe_df["Status"].value_counts())

# -------------- Feature Importance --------------

elif page == "🧠 Feature Importance":
    st.header("Feature Importance (Random Forest)")
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_columns, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False)
    fig = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Model Feature Importances",
        labels={"Importance": "Importance", "Feature": "Feature"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(fi_df)

    st.info("High importance means the model relied heavily on this feature for its decisions. Feature order may vary with retraining.")


