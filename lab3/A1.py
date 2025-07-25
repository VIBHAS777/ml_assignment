import pandas as pd
import numpy as np

df = pd.read_excel("AirQualityUCI.xlsx")

features = df.drop(columns=['Date', 'Time'])

features = features.dropna()

class1 = features[features['CO(GT)'] > 2.0]
class2 = features[features['CO(GT)'] <= 2.0]

class1_features = class1.drop(columns=['CO(GT)'])
class2_features = class2.drop(columns=['CO(GT)'])

mean1 = np.mean(class1_features.values, axis=0)
mean2 = np.mean(class2_features.values, axis=0)

spread1 = np.std(class1_features.values, axis=0)
spread2 = np.std(class2_features.values, axis=0)

interclass_distance = np.linalg.norm(mean1 - mean2)

print("Class 1 Mean (centroid):", mean1)
print("Class 2 Mean (centroid):", mean2)
print("\nClass 1 Spread (std):", spread1)
print("Class 2 Spread (std):", spread2)
print("\nInterclass Distance between Centroids:", interclass_distance)
