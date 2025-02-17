import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error,accuracy_score

def run():
    # Load the trained model and validation data
    model = joblib.load("models/model.pkl")
    X_val = joblib.load("data/X_val.pkl")
    Y_val = joblib.load("data/Y_val.pkl")

    # Make predictions
    predictions = model.predict(X_val)

    # Print the predictions and calculate Mean Absolute Error (MAE)
    print(f"Predictions: {predictions}")
    mae = mean_absolute_error(Y_val, predictions)

    print(f"Mean Absolute Error: {mae}")
    accuracy=accuracy_score(Y_val,predictions)
    print(f"the accurancy is equal to : {accuracy}")

