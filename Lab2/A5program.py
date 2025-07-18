import pandas as pd
import numpy as np

def calculate_jc_smc(vec1, vec2):
    f11 = np.sum(vec1 & vec2)
    f00 = np.sum(~vec1 & ~vec2)
    f01 = np.sum(~vec1 & vec2)
    f10 = np.sum(vec1 & ~vec2)

    jc = f11 / (f01 + f10 + f11)
    smc = (f11 + f00) / (f00 + f01 + f10 + f11)
    return jc, smc

file_path = "Lab Session Data.xlsx"
thyroid_df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

binary_df = thyroid_df.select_dtypes(include=[np.number]).astype('bool')
vec1 = binary_df.iloc[0]
vec2 = binary_df.iloc[1]

jc, smc = calculate_jc_smc(vec1, vec2)
print(f"Jaccard Coefficient: {jc:}")
print(f"Simple Matching Coefficient: {smc:}")
