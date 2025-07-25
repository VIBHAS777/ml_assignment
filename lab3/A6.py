import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_excel("AirQualityUCI.xlsx")
data['CO(GT)'] = (data['CO(GT)'] > 2).astype(int)

features = data[['PT08.S1(CO)', 'PT08.S2(NMHC)']]
target = data['CO(GT)']

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

test_accuracy = model.score(X_test, y_test)
print("Accuracy using .score():", test_accuracy)

