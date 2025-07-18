
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cosine

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
df = df.dropna()

for col in df.select_dtypes(include='object'):
    df[col] = df[col].astype(str)
    df[col] = LabelEncoder().fit_transform(df[col])

print("A4:")
print(df.describe())

v1 = (df.iloc[0] != 0).astype(int)
v2 = (df.iloc[1] != 0).astype(int)
f11 = np.sum((v1 == 1) & (v2 == 1))
f00 = np.sum((v1 == 0) & (v2 == 0))
f10 = np.sum((v1 == 1) & (v2 == 0))
f01 = np.sum((v1 == 0) & (v2 == 1))
jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) else 0
smc = (f11 + f00) / (f11 + f00 + f10 + f01)
print("\nA5:\nJaccard:", jc, "\nSMC:", smc)

cs = 1 - cosine(df.iloc[0], df.iloc[1])
print("\nA6:\nCosine Similarity:", cs)
