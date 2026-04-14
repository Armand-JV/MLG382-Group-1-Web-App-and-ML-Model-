import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Diabetes_and_LifeStyle_Dataset_.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.joblib"
LABEL_ENCODER_PATH = PROCESSED_DIR / "label_encoder.joblib"


def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    logging.info(f"Loaded dataset: {df.shape}")
    return df


def clean_data(df):
    df = df.copy()

    # Standardize column names FIRST
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Remove duplicates
    df = df.drop_duplicates()
    logging.info(f"After duplicate removal: {df.shape}")

    # Missing values
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def engineer_features(df):
    df = df.copy()

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 30, 45, 60, 100],
            labels=["Young", "Middle-aged", "Senior", "Elderly"]
        )

    if "bmi" in df.columns:
        df["bmi_category"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 25, 30, 100],
            labels=["Underweight", "Normal", "Overweight", "Obese"]
        )

    if "systolic_bp" in df.columns and "diastolic_bp" in df.columns:
        df["bp_category"] = "Normal"

        df.loc[
            (df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90),
            "bp_category"
        ] = "Hypertension"

        df.loc[
            ((df["systolic_bp"] >= 120) & (df["systolic_bp"] < 140)) |
            ((df["diastolic_bp"] >= 80) & (df["diastolic_bp"] < 90)),
            "bp_category"
        ] = "Prehypertension"

    logging.info("Feature engineering completed")
    return df


def remove_outliers_iqr(df, numeric_cols):
    df = df.copy()

    for col in numeric_cols:
        if col not in df.columns:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


def split_features_targets(df):
    target_col = "diabetes_stage"
    binary_col = "diagnosed_diabetes"

    if target_col not in df.columns:
        raise ValueError("Missing target column: diabetes_stage")

    X = df.drop(columns=[target_col, binary_col], errors="ignore")
    y = df[target_col]
    y_binary = df[binary_col] if binary_col in df.columns else None

    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    return X, y, y_binary, num_features, cat_features


def build_preprocessor(num_features, cat_features):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_features),
            ("cat", categorical_pipe, cat_features)
        ]
    )



#Pipline
def run_pipeline(df):
    # Step 1: clean + feature engineering
    df = clean_data(df)
    df = engineer_features(df)

    # Step 2: split
    X, y, y_binary, num_features, cat_features = split_features_targets(df)

    # Step 3: encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Step 4: TRAIN/TEST SPLIT (IMPORTANT BEFORE OUTLIERS OR FITTING)
    X_train, X_test, y_train, y_test, yb_train, yb_test = train_test_split(
        X,
        y_encoded,
        y_binary,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Step 5: OUTLIER REMOVAL ONLY ON TRAIN SET (FIX LEAKAGE)
    X_train = remove_outliers_iqr(X_train, num_features)

    # Align y after outlier removal
    y_train = y_train[:len(X_train)]
    if yb_train is not None:
        yb_train = yb_train[:len(X_train)]

    # Step 6: build preprocessing pipeline
    preprocessor = build_preprocessor(num_features, cat_features)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Step 7: correct feature names (FIXED)
    feature_names = preprocessor.get_feature_names_out()

    logging.info(f"Train shape: {X_train_processed.shape}")
    logging.info(f"Test shape: {X_test_processed.shape}")

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "y_binary_train": yb_train,
        "y_binary_test": yb_test,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "feature_names": feature_names
    }


def save_artifacts(result):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    np.save(PROCESSED_DIR / "X_train.npy", result["X_train"])
    np.save(PROCESSED_DIR / "X_test.npy", result["X_test"])
    np.save(PROCESSED_DIR / "y_train.npy", result["y_train"])
    np.save(PROCESSED_DIR / "y_test.npy", result["y_test"])

    joblib.dump(result["preprocessor"], PREPROCESSOR_PATH)
    joblib.dump(result["label_encoder"], LABEL_ENCODER_PATH)

    pd.DataFrame(result["feature_names"]).to_csv(
        PROCESSED_DIR / "feature_names.csv",
        index=False
    )

    logging.info(f"Artifacts saved in {PROCESSED_DIR}")


if __name__ == "__main__":
    df = load_data()
    result = run_pipeline(df)
    save_artifacts(result)
    logging.info("Data preparation complete!")