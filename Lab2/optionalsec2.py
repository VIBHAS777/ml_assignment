import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

xls = pd.ExcelFile("Lab Session Data.xlsx")
df_thyroid = xls.parse("thyroid0387_UCI")

df_sample = df_thyroid.dropna().sample(20, random_state=42)

numeric_data = df_sample.select_dtypes(include=['int64', 'float64'])

binary_data = numeric_data.applymap(lambda x: 1 if x != 0 else 0)

vec1 = binary_data.iloc[0]
vec2 = binary_data.iloc[1]

f11 = np.sum((vec1 == 1) & (vec2 == 1))
f00 = np.sum((vec1 == 0) & (vec2 == 0))
f10 = np.sum((vec1 == 1) & (vec2 == 0))
f01 = np.sum((vec1 == 0) & (vec2 == 1))

denominator_jc = f01 + f10 + f11
denominator_smc = f11 + f00 + f01 + f10

jaccard = f11 / denominator_jc if denominator_jc != 0 else 0
smc = (f11 + f00) / denominator_smc if denominator_smc != 0 else 0

vec1_real = numeric_data.iloc[0]
vec2_real = numeric_data.iloc[1]
cos_sim = cosine_similarity([vec1_real], [vec2_real])[0][0]

print("jaccard Coefficient:", jaccard)
print("simple matching Coefficient:", smc)
print("cosine Similarity:", cos_sim)

