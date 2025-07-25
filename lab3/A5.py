
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_excel("AirQualityUCI.xlsx")

df['CO(GT)'] = df['CO(GT)'].apply(lambda x: 1 if x > 2 else 0)

X = df[['PT08.S1(CO)', 'PT08.S2(NMHC)']]
y = df['CO(GT)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

y_pred = neigh.predict(X_test)

print("First 10 predictions on test set:", y_pred[:10])
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
