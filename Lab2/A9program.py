import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

file_path = "Lab Session Data.xlsx"
thyroid_df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

normalized_df = normalize(thyroid_df.copy())
print("Normalized sample:\n", normalized_df.head())
