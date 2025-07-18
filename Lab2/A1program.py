import pandas as pd
import numpy as np
from numpy.linalg import matrix_rank, pinv

# Load and clean data
xls = pd.ExcelFile("Lab Session Data.xlsx")
df = xls.parse('Purchase data')
df_clean = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']].dropna()

A = df_clean[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].to_numpy()
C = df_clean[['Payment (Rs)']].to_numpy()

print("dimensionality:", A.shape[1])
print("no of vectors:", A.shape[0])
print("Rank of A:", matrix_rank(A))

X = pinv(A) @ C
print("Cost per product (Candy, Mango, Milk):", X.flatten())

