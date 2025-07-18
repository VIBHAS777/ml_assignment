import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statistics

xls = pd.ExcelFile("Lab Session Data.xlsx")
df_purchase = xls.parse("Purchase data")

df_purchase_clean = df_purchase[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']].dropna()

A_full = df_purchase_clean.iloc[:, 0:3].values
C_full = df_purchase_clean.iloc[:, 3].values

A_sq1 = A_full[0:3]
C_sq1 = C_full[0:3]

A_sq2 = A_full[3:6]
C_sq2 = C_full[3:6]

def pseudo_inverse_solution(A, C):
    return np.dot(np.linalg.pinv(A), C)

X_full = pseudo_inverse_solution(A_full, C_full)
X_sq1 = pseudo_inverse_solution(A_sq1, C_sq1)
X_sq2 = pseudo_inverse_solution(A_sq2, C_sq2)

print("X vector from full matrix:\n", X_full)
print("X vector from square matrix 1:\n", X_sq1)
print("X vector from square matrix 2:\n", X_sq2)

labels = ['RICH' if val > 200 else 'POOR' for val in C_full]
classifier = LogisticRegression()
classifier.fit(A_full, labels)
accuracy = classifier.score(A_full, labels)

print("Classifier Accuracy (RICH/POOR):", accuracy)

df_stock = xls.parse("IRCTC Stock Price")
prices = df_stock.iloc[:, 3].dropna()
mean_price = statistics.mean(prices)
variance_price = statistics.variance(prices)

print("IRCTC Price Mean:", mean_price)
print("IRCTC Price Variance:", variance_price)

