import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_excel("AirQualityUCI.xlsx")
df['CO(GT)'] = df['CO(GT)'].apply(lambda x: 1 if x > 2 else 0)

X = df[['PT08.S1(CO)', 'PT08.S2(NMHC)']]
y = df['CO(GT)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k_values = list(range(1, 12))  # from 1 to 11
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    accuracies.append(acc)
    print(f"k={k}, Accuracy={acc:.4f}")

plt.plot(k_values, accuracies, marker='o', color='red')
plt.title('k-NN Accuracy vs k')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy on Test Set')
plt.grid(True)
plt.xticks(k_values)
plt.show()

