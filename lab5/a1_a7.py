import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split

# Load and prepare the dataset
def load_and_preprocess(filepath):
    df = pd.read_excel(filepath)
    df = df.select_dtypes(include=[np.number])  # keep only numeric columns
    df = df.dropna()  # remove rows with missing values
    return df

# Train a linear regression model
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate regression performance
def evaluate_regression(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, rmse, mape, r2

# Run k-means clustering
def perform_kmeans(X, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    return kmeans

# Calculate clustering metrics
def clustering_metrics(X, labels):
    silhouette = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_index = davies_bouldin_score(X, labels)
    return silhouette, ch_score, db_index

# Plot clustering evaluation metrics for different k values
def plot_clustering_metrics(X, k_values):
    silhouette_scores, ch_scores, db_scores = [], [], []

    for k in k_values:
        kmeans = perform_kmeans(X, k)
        s, ch, db = clustering_metrics(X, kmeans.labels_)
        silhouette_scores.append(s)
        ch_scores.append(ch)
        db_scores.append(db)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, marker='o', label="Silhouette Score")
    plt.plot(k_values, ch_scores, marker='s', label="Calinski-Harabasz Score")
    plt.plot(k_values, db_scores, marker='^', label="Davies-Bouldin Index")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Score")
    plt.title("Clustering Metrics vs k")
    plt.legend()
    plt.grid(True)
    plt.show()

# Elbow plot for k-means
def elbow_plot(X, k_values):
    distortions = []
    for k in k_values:
        kmeans = perform_kmeans(X, k)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, distortions, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)
    plt.show()

def main():
    df = load_and_preprocess("AirQualityUCI.xlsx")

    # Use the first numeric column as target
    target_column = df.columns[0]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # A1 - Single attribute regression
    X_train_single = X_train.iloc[:, [0]].values
    X_test_single = X_test.iloc[:, [0]].values
    model_single = train_linear_regression(X_train_single, y_train)
    print("A1: Linear Regression model trained using a single attribute.")

    # A2 - Metrics for single attribute regression
    train_metrics_single = evaluate_regression(model_single, X_train_single, y_train)
    test_metrics_single = evaluate_regression(model_single, X_test_single, y_test)
    print("\nA2: Performance metrics for single attribute regression")
    print(f"Train -> MSE: {train_metrics_single[0]:.4f}, RMSE: {train_metrics_single[1]:.4f}, "
          f"MAPE: {train_metrics_single[2]:.4f}, R2: {train_metrics_single[3]:.4f}")
    print(f"Test  -> MSE: {test_metrics_single[0]:.4f}, RMSE: {test_metrics_single[1]:.4f}, "
          f"MAPE: {test_metrics_single[2]:.4f}, R2: {test_metrics_single[3]:.4f}")

    # A3 - Multiple attributes regression
    model_multi = train_linear_regression(X_train, y_train)
    train_metrics_multi = evaluate_regression(model_multi, X_train, y_train)
    test_metrics_multi = evaluate_regression(model_multi, X_test, y_test)
    print("\nA3: Performance metrics for regression using all attributes")
    print(f"Train -> MSE: {train_metrics_multi[0]:.4f}, RMSE: {train_metrics_multi[1]:.4f}, "
          f"MAPE: {train_metrics_multi[2]:.4f}, R2: {train_metrics_multi[3]:.4f}")
    print(f"Test  -> MSE: {test_metrics_multi[0]:.4f}, RMSE: {test_metrics_multi[1]:.4f}, "
          f"MAPE: {test_metrics_multi[2]:.4f}, R2: {test_metrics_multi[3]:.4f}")

    # A4 - K-means clustering with k=2
    kmeans_2 = perform_kmeans(X_train, 2)
    print("\nA4: K-Means clustering with k=2 completed.")
    print("Cluster Centers:\n", kmeans_2.cluster_centers_)
    print("Cluster Labels:\n", kmeans_2.labels_)

    # A5 - Clustering metrics for k=2
    silhouette, ch_score, db_index = clustering_metrics(X_train, kmeans_2.labels_)
    print("\nA5: Clustering evaluation metrics for k=2")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Calinski-Harabasz Score: {ch_score:.4f}")
    print(f"Davies-Bouldin Index: {db_index:.4f}")

    # A6 - Clustering metrics for multiple k values
    k_values = range(2, 8)
    print("\nA6: Plotting clustering metrics for different k values...")
    plot_clustering_metrics(X_train, k_values)

    # A7 - Elbow plot
    print("\nA7: Generating elbow plot to determine optimal k...")
    elbow_plot(X_train, k_values)

if __name__ == "__main__":
    main()
