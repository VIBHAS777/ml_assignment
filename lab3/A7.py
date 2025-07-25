import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_excel("AirQualityUCI.xlsx")
data['CO(GT)'] = (data['CO(GT)'] > 2).astype(int)

features = data[['PT08.S1(CO)', 'PT08.S2(NMHC)']]
labels = data['CO(GT)']

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
print("First 10 predictions on test set:", predictions[:10])

first_sample = X_test.iloc[[0]]
first_pred = knn.predict(first_sample)
print("Prediction for first test vector:", first_pred[0])

