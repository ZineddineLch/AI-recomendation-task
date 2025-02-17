import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,accuracy_score

def run():
    # Load dataset
    df = pd.read_csv("data/book_recommendation_processed.csv")

    # Features and target variable
    X = df.drop(columns=["Recommendations"])
    y = df["Recommendations"]

    # Split data into training and testing
    X_train_val, X_test, Y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, y_val = train_test_split(X_train_val, Y_train_val, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Predicting the values
    y_pred = model.predict(X_test)

    # Print cost function (mean absolute error)
    print(f"Mean Absolute Error: ${mean_absolute_error(y_test, y_pred):.2f}")

    # Save the trained model and validation data
    joblib.dump(model, "models/model.pkl")
    joblib.dump(X_val, "data/X_val.pkl")
    joblib.dump(y_val, "data/Y_val.pkl")

    print("Training completed and model saved.")
