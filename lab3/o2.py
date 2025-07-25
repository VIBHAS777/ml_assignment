import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings

df = pd.read_excel("AirQualityUCI.xlsx")

df.replace(-200, pd.NA, inplace=True)
df.dropna(inplace=True)

features = ['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
            'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
            'PT08.S5(O3)', 'T', 'RH', 'AH']

X = df[features]

def co_label(value):
    if value <= 1.5:
        return 0
    elif value <= 3.0:
        return 1
    else:
        return 2

y = df['CO(GT)'].apply(co_label)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

metrics = ['euclidean', 'manhattan', 'chebyshev']
print("Accuracy with different distance metrics:")

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=3, metric=metric)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{metric.capitalize()} Distance: {acc:.4f}")

