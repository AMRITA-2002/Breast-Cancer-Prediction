import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("data\Breast_Cancer.csv")


def safe_label_encode(df, cols_to_encode):
    label_encoders = {}
    for col in cols_to_encode:
        df[col] = (
            df[col].astype(str).str.strip().replace(["nan", "NaN", "", None], "Unknown")
        )
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders


def make_all_features_numeric(X):
    # After all feature selection, guarantee all features are numeric
    # This will catch any missed mixed-type or weird text columns
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            print(f"Encoding remaining non-numeric column: {col}")
            X[col] = (
                X[col]
                .astype(str)
                .str.strip()
                .replace(["nan", "NaN", "", None], "Unknown")
            )
            X[col] = LabelEncoder().fit_transform(X[col])
    # Final check:
    assert all([np.issubdtype(dt, np.number) for dt in X.dtypes]), (
        "Some columns are still not numeric!"
    )
    return X


def prepare_data():
    df = pd.read_csv("data\Breast_Cancer.csv")
    df.columns = df.columns.str.strip()
    print(f"Initial Dataset Shape: {df.shape}")

    # Fill all missing
    df = df.fillna("Unknown")

    # Specify all possible categorical columns
    categorical_cols = [
        col
        for col in [
            "Race",
            "Marital Status",
            "T Stage",
            "N Stage",
            "6th Stage",
            "differentiate",
            "A Stage",
            "Estrogen Status",
            "Progesterone Status",
            "Grade",
        ]
        if col in df.columns
    ]

    # Label Encode class columns except the target
    df, label_encoders = safe_label_encode(df, categorical_cols)

    # Target encoding
    target_encoder = LabelEncoder()
    df["Status_encoded"] = target_encoder.fit_transform(
        df["Status"].astype(str).str.strip()
    )

    # Select features (edit as needed per your actual dataset structure)
    feature_columns = []
    for col in [
        "Age",
        "Race",
        "Marital Status",
        "T Stage",
        "N Stage",
        "6th Stage",
        "differentiate",
        "Grade",
        "A Stage",
        "Tumor Size",
        "Estrogen Status",
        "Progesterone Status",
        "Regional Node Examined",
        "Reginol Node Positive",
        "Survival Months",
    ]:
        if col in df.columns:
            feature_columns.append(col)
    print(f"Features used: {feature_columns}")

    # Make sure all numeric-ish columns are numbers, not strings
    for col in [
        "Age",
        "Tumor Size",
        "Regional Node Examined",
        "Reginol Node Positive",
        "Survival Months",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    X = df[feature_columns].copy()
    y = df["Status_encoded"]

    # Guarantee all features numeric
    X = make_all_features_numeric(X)
    print("All features are now numeric.")

    return X, y, label_encoders, target_encoder, feature_columns


def train_model():
    X, y, label_encoders, target_encoder, feature_columns = prepare_data()
    print("Final dtype check:\n", X.dtypes)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Test accuracy: {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

    # Save all required objects
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/breast_cancer_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    joblib.dump(target_encoder, "models/target_encoder.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")
    print("All model objects saved under './models/'")


if __name__ == "__main__":
    print("=========== Starting Model Training ===========")
    try:
        train_model()
        print("=========== Training SUCCESS! ===========")
    except Exception as e:
        print("=========== ERROR ===========")
        import traceback

        traceback.print_exc()
