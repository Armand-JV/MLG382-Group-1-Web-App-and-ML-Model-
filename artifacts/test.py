import joblib 
import pandas as pd

kmeans_model = joblib.load("kmeans_model.joblib")


random_forest_model = joblib.load("random_forest_best_model.pkl")

xgb_model = joblib.load("xgboost_best_model.pkl")

label_encoder = joblib.load("label_encoder.pkl")

print("Model classes:", xgb_model.classes_)
print("Label encoder classes:", label_encoder.classes_)
