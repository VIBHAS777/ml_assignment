import pandas as pd
import numpy as np
from numpy.linalg import matrix_rank, pinv

# Load data
xls = pd.ExcelFile("Lab_Session_Data.xlsx")
df = xls.parse('Purchase data')
df_clean = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']].dropna()

# Create matrices A and C
A = df_clean[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].to_numpy()
C = df_clean[['Payment (Rs)']].to_numpy()

# Outputs
print("Dimensionality:", A.shape[1])
print("Number of vectors:", A.shape[0])
print("Rank of A:", matrix_rank(A))

# Find cost of each product using pseudo-inverse
X = pinv(A) @ C
print("Cost per product (Candy, Mango, Milk):", X.flatten())
