import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import lime.lime_tabular
import shap

# A1: Correlation analysis with heatmap
def compute_correlation_heatmap(data, exclude_cols=['Date', 'Time']):
    features = [col for col in data.columns if col not in exclude_cols]
    corr_matrix = data[features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()
    return corr_matrix

# Utility: Cleanup and target definition
def preprocess_data(data):
    # Remove timestamp columns
    data = data.drop(['Date', 'Time'], axis=1)
    data.replace(-200, np.nan, inplace=True)
    data = data.dropna()
    target = (data['AH'] > data['AH'].median()).astype(int)
    features = data.drop('AH', axis=1)
    return features, target

# A2/A3: PCA reduction and model performance
def pca_and_model(features, target, variance=None, model_cls=LogisticRegression, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=X_scaled.shape[1])
    X_pca = pca.fit_transform(X_scaled)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    if variance:
        n_components = np.where(cumsum >= variance)[0][0] + 1
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
    else:
        n_components = X_scaled.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X_pca, target, test_size=0.2, random_state=random_state)
    model = model_cls(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    return acc, clf_report, n_components

# A4: Sequential Feature Selection experiment
def sequential_selector_and_model(features, target, direction='forward', model_cls=LogisticRegression, n_features=5, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=random_state)
    model = model_cls(max_iter=1000, random_state=random_state)
    selector = SequentialFeatureSelector(model, n_features_to_select=n_features, direction=direction)
    selector.fit(X_train, y_train)
    X_selected = selector.transform(X_scaled)
    X_sel_train, X_sel_test, y_sel_train, y_sel_test = train_test_split(X_selected, target, test_size=0.2, random_state=random_state)
    model.fit(X_sel_train, y_sel_train)
    y_pred = model.predict(X_sel_test)
    acc = accuracy_score(y_sel_test, y_pred)
    clf_report = classification_report(y_sel_test, y_pred)
    selected_indices = selector.get_support(indices=True)
    selected_features = [features.columns[i] for i in selected_indices]
    return acc, clf_report, selected_features

# A5: LIME and SHAP explainability
def explain_with_lime_shap(features, target, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=random_state)
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)

    # LIME
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=features.columns, class_names=['Low', 'High'], discretize_continuous=True)
    idx = np.random.randint(0, X_test.shape[0])
    lime_exp = explainer_lime.explain_instance(X_test[idx], model.predict_proba, num_features=8)
    lime_exp.show_in_notebook(show_table=True)
    
    # SHAP
    explainer_shap = shap.Explainer(model, X_train)
    shap_values = explainer_shap(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=features.columns)

    return lime_exp, shap_values

if __name__ == "__main__":
    data = pd.read_excel("AirQualityUCI.xlsx")
    
    # A1. Correlation Heatmap
    print("\nA1: Correlation Matrix Heatmap")
    corr_matrix = compute_correlation_heatmap(data)

    # Preprocess
    features, target = preprocess_data(data)
    
    # A2. PCA (99%)
    print("\nA2: PCA Reduction (99% Variance)")
    acc_99, report_99, ncomp_99 = pca_and_model(features, target, variance=0.99)
    print(f"Accuracy (99% variance, {ncomp_99} components):", acc_99)
    print(report_99)
    
    # A3. PCA (95%)
    print("\nA3: PCA Reduction (95% Variance)")
    acc_95, report_95, ncomp_95 = pca_and_model(features, target, variance=0.95)
    print(f"Accuracy (95% variance, {ncomp_95} components):", acc_95)
    print(report_95)

    # A4. Sequential Feature Selector
    print("\nA4: Sequential Feature Selection (5 features)")
    acc_sfs, report_sfs, sel_feats = sequential_selector_and_model(features, target, n_features=5)
    print(f"Accuracy (SFS, selected features: {sel_feats}):", acc_sfs)
    print(report_sfs)

    # A5. LIME & SHAP Explanations
    print("\nA5: LIME and SHAP explanations")
    lime_exp, shap_values = explain_with_lime_shap(features, target)
    print("LIME and SHAP explanations displayed.")

