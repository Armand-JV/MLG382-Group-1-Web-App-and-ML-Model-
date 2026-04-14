import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.joblib"
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Diabetes_and_LifeStyle_Dataset_.csv"

CLUSTER_MODEL_PATH = MODEL_DIR / "kmeans_model.joblib"
CLUSTER_OUTPUT_PATH = PROCESSED_DIR / "clustered_data.csv"


#Loading data and preprocessor
def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    return df
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


def basic_clean(df):
    df = df.copy()

    # Standardize column names (must match preprocessing pipeline)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Fill missing values (safe fallback for clustering step)
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def feature_engineering(df):
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

    return df


#Pipeline
def run_kmeans():

    # 1. Load data + preprocessing object
    df = load_data()
    preprocessor = load_preprocessor()

    # 2. Clean + feature engineering
    df = basic_clean(df)
    df = feature_engineering(df)

    # 3. Drop target columns (unsupervised learning)
    drop_cols = ["diabetes_stage", "diagnosed_diabetes"]
    X = df.drop(columns=drop_cols, errors="ignore")

    # 4. Transform data using EXISTING preprocessor
    X_processed = preprocessor.transform(X)

    # 5. Apply KMeans clustering
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init="auto"
    )

    clusters = kmeans.fit_predict(X_processed)

    # 6. Attach clusters to dataset
    df["cluster"] = clusters

    # 7. Save model + output
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    joblib.dump(kmeans, CLUSTER_MODEL_PATH)

    df.to_csv(CLUSTER_OUTPUT_PATH, index=False)

    # 8. Basic insights
    print("\nCluster distribution:")
    print(df["cluster"].value_counts())

    print(f"\nModel saved to: {CLUSTER_MODEL_PATH}")
    print(f"Clustered dataset saved to: {CLUSTER_OUTPUT_PATH}")

    return df, kmeans

#Run
if __name__ == "__main__":
    run_kmeans()