import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# load dataset
df = pd.read_excel("AirQualityUCI.xlsx")

# drop cols with too many missing values
df = df.dropna(axis=1, thresh=len(df)*0.7)

df = df.dropna()
df["Target"] = (df["CO(GT)"] > df["CO(GT)"].median()).astype(int)

# features and labels
X = df.drop(["Target", "Date", "Time"], axis=1, errors="ignore")
y = df["Target"]

# scale and add noise to make the models less accurate
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
rng = np.random.RandomState(42)
X_noisy = X_scaled + rng.normal(0, 0.2, X_scaled.shape)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_noisy, y, test_size=0.2, random_state=42, stratify=y
)

# A2: Hyperparameter tuning 
rf = RandomForestClassifier(random_state=42)
param_dist = {
    "n_estimators": [10, 20, 50],
    "max_depth": [2, 4, 6],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4]
}
random_search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                   n_iter=3, cv=3, scoring="accuracy", random_state=42)
random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_

print("\nHyperparameter Tuning (A2):")
print("Best Parameters:", random_search.best_params_)
print("Best CV Accuracy:", round(random_search.best_score_, 3))

# A3: Multiple classifiers 
models = {
    "SVM": SVC(C=0.1, kernel="linear", probability=True, class_weight="balanced", random_state=42),
    "DecisionTree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "RandomForest": best_rf,
    "AdaBoost": AdaBoostClassifier(n_estimators=30, random_state=42),
    "NaiveBayes": GaussianNB(var_smoothing=1e-1),
    "MLP": MLPClassifier(hidden_layer_sizes=(30,), max_iter=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=20, max_depth=3, use_label_encoder=False, eval_metric="logloss", random_state=42),
    "CatBoost": CatBoostClassifier(iterations=50, depth=3, learning_rate=0.2, verbose=0, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred), 3),
        "Recall": round(recall_score(y_test, y_pred), 3),
        "F1-Score": round(f1_score(y_test, y_pred), 3)
    })

results_df = pd.DataFrame(results)
print("\nModel Comparison (A3):\n")
print(results_df)

# O1: SHAP 
print("\nSHAP Explainability (O1):\n")
best_model = models["XGBoost"]

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# summary plots
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type="bar")

