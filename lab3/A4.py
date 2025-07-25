import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel("AirQualityUCI.xlsx")

df = df.dropna()
df = df.select_dtypes(include=['float64', 'int64'])

X = df[['CO(GT)', 'NOx(GT)']]

y = (df['CO(GT)'] > 2).astype(int)

print("Class distribution:\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

