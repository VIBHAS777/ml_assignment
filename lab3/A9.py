
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_excel("AirQualityUCI.xlsx")
data['CO(GT)'] = (data['CO(GT)'] > 2).astype(int)

features = data[['PT08.S1(CO)', 'PT08.S2(NMHC)']]
labels = data['CO(GT)']

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

print("\nConfusion Matrix - Training:")
print(confusion_matrix(y_train, train_predictions))

print("\nClassification Report - Training:")
print(classification_report(y_train, train_predictions))

print("\nConfusion Matrix - Testing:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification Report - Testing:")
print(classification_report(y_test, test_predictions))
