import pandas as pd
import numpy as np

def impute_data(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        if ((df[col] - df[col].mean()).abs() > 3 * df[col].std()).any():
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

file_path = "Lab Session Data.xlsx"
thyroid_df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

imputed_df = impute_data(thyroid_df.copy())
print("Missing values after imputation:\n", imputed_df.isnull().sum())
