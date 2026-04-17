import pandas as pd
import joblib as joblib
import pickle


#load models 
xgb_model     = joblib.load("artifacts/xgboost_best_model.pkl")
preprocessor  = joblib.load("data/processed/preprocessor.joblib")
label_encoder = joblib.load("data/processed/label_encoder.joblib")
kmeans_model  = joblib.load("artifacts/kmeans_model.joblib")

print("Model classes:", xgb_model.classes_)
print("Label encoder classes:", label_encoder.classes_)