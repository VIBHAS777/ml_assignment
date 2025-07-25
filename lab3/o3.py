import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

data = pd.read_excel("AirQualityUCI.xlsx")
data = data.dropna()
numeric_data = data.select_dtypes(include=np.number)

numeric_data['Label'] = (numeric_data['CO(GT)'] > 2).astype(int)

features = numeric_data.drop(columns='Label')
labels = numeric_data['Label']

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

probabilities = model.predict_proba(X_test_scaled)[:, 1]

fpr_vals, tpr_vals, _ = roc_curve(y_test, probabilities)
auc_score = auc(fpr_vals, tpr_vals)

plt.figure(figsize=(8, 6))
plt.plot(fpr_vals, tpr_vals, color='blue', label=f'k-NN (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - k-NN Classifier')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

