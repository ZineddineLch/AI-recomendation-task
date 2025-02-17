import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def run():
    # Load the dataset
    df = pd.read_csv("data/book_recommendation.csv")

    # Handle missing values by filling with the mean
    # Filter numeric columns only
    numeric_columns = df.select_dtypes(include=['number'])
    
    # Fill missing values with mean of numeric columns only
    df[numeric_columns.columns] = numeric_columns.fillna(numeric_columns.mean())

    # Encoding categorical columns using LabelEncoder
    encoder = LabelEncoder()
    
    # Convert categorical columns to numeric
    for column in df.select_dtypes(include=[object]).columns:
        df[column] = encoder.fit_transform(df[column])

    # Save the processed data
    df.to_csv("data/book_recommendation_processed.csv", index=False)  # Save to a new file to avoid overwriting the original

    print("Data preprocessing completed.")