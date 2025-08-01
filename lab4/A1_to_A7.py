import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

def knn_classification_metrics(X, y, k=3):
    # Split data and train kNN
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_tr, y_tr)

    # Predictions
    pred_tr = clf.predict(X_tr)
    pred_ts = clf.predict(X_ts)

    # Confusion matrix and reports
    cm_tr = confusion_matrix(y_tr, pred_tr)
    cm_ts = confusion_matrix(y_ts, pred_ts)
    rpt_tr = classification_report(y_tr, pred_tr, output_dict=True)
    rpt_ts = classification_report(y_ts, pred_ts, output_dict=True)

    return cm_tr, cm_ts, rpt_tr, rpt_ts

def regression_scores(X, y):
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)

    # Metrics
    mse = mean_squared_error(y, pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y, pred)
    r2 = r2_score(y, pred)

    return mse, rmse, mape, r2

def create_simple_dataset():
    # Random 2D points and assign class
    np.random.seed(42)
    data = np.random.randint(1, 11, size=(20, 2))
    labels = np.array([0 if (pt[0] + pt[1]) < 10 else 1 for pt in data])
    return data, labels

def predict_grid(X, y, k=3):
    # Train and predict over grid
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)
    xx, yy = np.meshgrid(np.arange(0, 10.1, 0.1), np.arange(0, 10.1, 0.1))
    grid_pts = np.c_[xx.ravel(), yy.ravel()]
    preds = clf.predict(grid_pts)
    return xx, yy, preds.reshape(xx.shape)

def draw_boundaries(X, y, k_vals):
    # Plot boundaries for different k
    for k in k_vals:
        xx, yy, zz = predict_grid(X, y, k)
        plt.figure(figsize=(6, 5))
        plt.contourf(xx, yy, zz, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k')
        plt.title(f"Decision Boundary (k = {k})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()

def classify_air_quality(df, col1, col2, target_col, threshold):
    # Use two features and threshold to form binary classes
    df_clean = df[[col1, col2, target_col]].dropna()
    X = df_clean[[col1, col2]].values
    y = (df_clean[target_col] > threshold).astype(int).values
    return X, y

def search_best_k(X, y):
    # Find best k using cross-validation
    params = {'n_neighbors': list(range(1, 21))}
    clf = KNeighborsClassifier()
    grid = GridSearchCV(clf, params, cv=5, scoring='accuracy')
    grid.fit(X, y)
    return grid.best_params_, grid.best_score_

if __name__ == "__main__":
    print("Lab 3 - Classification Models\n")

    # A1
    print("A1: kNN Metrics...")
    try:
        df_air = pd.read_excel("AirQualityUCI.xlsx").dropna()
        X1 = df_air[["C6H6(GT)", "T"]].values
        y1 = (df_air["CO(GT)"] > 2.5).astype(int).values
        cm_tr, cm_ts, rpt_tr, rpt_ts = knn_classification_metrics(X1, y1, k=3)

        print("Train Confusion Matrix:\n", cm_tr)
        print("\nTest Confusion Matrix:\n", cm_ts)
        print(f"\nTrain - Precision: {rpt_tr['weighted avg']['precision']:.3f}, Recall: {rpt_tr['weighted avg']['recall']:.3f}, F1: {rpt_tr['weighted avg']['f1-score']:.3f}")
        print(f"Test - Precision: {rpt_ts['weighted avg']['precision']:.3f}, Recall: {rpt_ts['weighted avg']['recall']:.3f}, F1: {rpt_ts['weighted avg']['f1-score']:.3f}")
    except Exception as e:
        print(f"Error in A1: {e}")

    # A2
    print("\nA2: Regression Evaluation...")
    try:
        df_lab = pd.read_excel("Lab_Session_Data.xlsx")
        df_lab = df_lab[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)"]].dropna()
        X2 = df_lab[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]]
        y2 = df_lab["Payment (Rs)"]
        mse, rmse, mape, r2 = regression_scores(X2, y2)

        print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.3f}, R²: {r2:.3f}")
    except Exception as e:
        print(f"Error in A2: {e}")

    # A3
    print("\nA3: Plotting Sample Dataset...")
    X3, y3 = create_simple_dataset()
    plt.figure(figsize=(6, 5))
    plt.scatter(X3[:, 0], X3[:, 1], c=y3, cmap=plt.cm.coolwarm, edgecolor='k')
    plt.title("Training Data (0: Blue, 1: Red)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

    # A4
    print("\nA4: Grid Prediction (k=3)...")
    xx, yy, zz = predict_grid(X3, y3, k=3)
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X3[:, 0], X3[:, 1], c=y3, cmap=plt.cm.coolwarm, edgecolor='k')
    plt.title("Test Grid Classification")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

    # A5
    print("\nA5: Plotting Boundaries...")
    draw_boundaries(X3, y3, k_vals=[1, 3, 5, 7, 15])

    # A6
    print("\nA6: Air Quality Classification...")
    try:
        X6, y6 = classify_air_quality(df_air, "PT08.S1(CO)", "T", "CO(GT)", 2.5)
        draw_boundaries(X6[:20], y6[:20], k_vals=[1, 3, 5])
    except Exception as e:
        print(f"Error in A6: {e}")

    # A7
    print("\nA7: Best k with Grid Search...")
    try:
        best_params, best_acc = search_best_k(X1, y1)
        print(f"Best k: {best_params['n_neighbors']}, Accuracy: {best_acc:.3f}")
    except Exception as e:
        print(f"Error in A7: {e}")
