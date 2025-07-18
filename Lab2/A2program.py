import numpy as np
import pandas as pd
from numpy import linalg

file_path = "Lab Session Data.xlsx"

purchase_data = pd.read_excel(file_path, sheet_name="Purchase data")

A = purchase_data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
C = purchase_data[["Payment (Rs)"]].values

dim_A = A.shape
vec_A = A.shape[1]
rank_A = linalg.matrix_rank(A)
pinv_A = linalg.pinv(A)
x = np.dot(pinv_A, C)

print(dim_A, vec_A, rank_A, x)
purchase_data["Category"] = np.where(purchase_data["Payment (Rs)"] > 200, "RICH", "POOR")
print(purchase_data[["Customer", "Category"]])
