import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    file_path = "Lab Session Data.xlsx"
    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
    return df

def identify_attribute_types(df):
    print("\nAttribute Types:")
    for col in df.columns:
        dtype = df[col].dtype
        unique_values = df[col].unique()
        if dtype == 'object':
            print(f"{col}: Categorical (Nominal/Ordinal) - Unique Values: {unique_values[:5]}")
        else:
            print(f"{col}: Numeric - Range: {df[col].min()} to {df[col].max()}")

def encode_categorical(df):
    label_encoded = df.copy()
    encoders = {}
    print("\nEncoding Categorical Columns:")
    for col in df.columns:
        if df[col].dtype == 'object':
            encoder = LabelEncoder()
            label_encoded[col] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder
            print(f"{col}: Label Encoded")
    return label_encoded, encoders

def check_missing_values(df):
    print("\nMissing Values in each column:")
    print(df.isnull().sum())

def detect_outliers(df):
    print("\nOutliers (Visual Representation):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols].plot(kind='box', figsize=(15, 6), subplots=True, layout=(1, len(numeric_cols)))
    plt.tight_layout()
    plt.show()

def mean_and_variance(df):
    print("\nMean and Standard Deviation for Numeric Columns:")
    numeric = df.select_dtypes(include=[np.number])
    for col in numeric.columns:
        print(f"{col}: Mean = {numeric[col].mean():.4f}, Std Dev = {numeric[col].std():.4f}")

def main():
    df = load_data()
    print("First 5 rows of data:\n", df.head())
    identify_attribute_types(df)
    check_missing_values(df)
    detect_outliers(df)
    mean_and_variance(df)
    encoded_df, encoders = encode_categorical(df)
    print("\nEncoded Data (First 5 rows):\n", encoded_df.head())

main()
