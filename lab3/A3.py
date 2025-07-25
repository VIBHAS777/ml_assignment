import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("AirQualityUCI.xlsx")

feature1 = 'CO(GT)'
feature2 = 'NOx(GT)'

df = df[[feature1, feature2]].dropna()

x = df.iloc[0].values
y = df.iloc[1].values

r_values = range(1, 11)
distances = []

for r in r_values:
    distance = np.power(np.sum(np.abs(x - y)**r), 1/r)
    distances.append(distance)

for r, d in zip(r_values, distances):
    print(f"Minkowski distance (r={r}): {d:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(r_values, distances, marker='o', color='blue')
plt.title(f"Minkowski Distance between 2 points\nFeatures: {feature1} & {feature2}")
plt.xlabel("r value")
plt.ylabel("Minkowski Distance")
plt.grid(True)
plt.tight_layout()
plt.show()
