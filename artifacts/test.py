import joblib 
import pandas as pd

kmeans_model = joblib.load("kmeans_model.joblib")
print(type(kmeans_model))

random_forest_model = joblib.load("random_forest_best_model.pkl")
print(type(random_forest_model))

xgboost_model = joblib.load("xgboost_best_model.pkl")
print(type(xgboost_model))

print("Models loaded successfully!")
print("KMeans Model:", kmeans_model)
print("Random Forest Model:", random_forest_model)
print("XGBoost Model:", xgboost_model)

print("Here are the feature importances for each model:")
print("Random Forest Feature Importances:", random_forest_model.feature_importances_)
print("XGBoost Feature Importances:", xgboost_model.feature_importances_)

print("KMeans Cluster Centers:", kmeans_model.cluster_centers_)

import pandas as pd

centers_df = pd.DataFrame(
    kmeans_model.cluster_centers_,
    columns=feature_names
)

print(centers_df)