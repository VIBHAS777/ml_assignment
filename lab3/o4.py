import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_excel("AirQualityUCI.xlsx", engine='openpyxl')

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

df.replace(-200, np.nan, inplace=True)

df.dropna(thresh=15, inplace=True)

df.drop(['Date', 'Time'], axis=1, inplace=True)

df = df.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
df = df.apply(pd.to_numeric, errors='coerce')

df.fillna(df.mean(numeric_only=True), inplace=True)


df['CO_Class'] = df['CO(GT)'].apply(lambda x: 1 if x > 2.0 else 0)

X = df.drop(['CO(GT)', 'CO_Class'], axis=1)
y = df['CO_Class'].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def custom_knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
        k_indices = np.argsort(distances)[:k]
        k_labels = [y_train[i] for i in k_indices]
        prediction = max(set(k_labels), key=k_labels.count)
        predictions.append(prediction)
    return np.array(predictions)

y_pred_custom = custom_knn_predict(X_train, list(y_train), X_test, k=3)
acc_custom = accuracy_score(y_test, y_pred_custom)
print(f"\nCustom k-NN Accuracy: {acc_custom:.4f}")

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_sklearn = knn.predict(X_test)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"Scikit-learn k-NN Accuracy: {acc_sklearn:.4f}")

