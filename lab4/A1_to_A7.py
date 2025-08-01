
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

def evaluate_knn_metrics(X, y, k=3):
    """
    A1: Evaluate confusion matrix and performance metrics for kNN classification
    Returns confusion matrices and classification reports for both training and test sets
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    report_train = classification_report(y_train, y_train_pred, output_dict=True)
    report_test = classification_report(y_test, y_test_pred, output_dict=True)
    return cm_train, cm_test, report_train, report_test

def regression_metrics(X, y):
    """
    A2: Calculate MSE, RMSE, MAPE and R2 scores for regression
    Returns regression performance metrics
    """
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, rmse, mape, r2

def generate_training_data():
    """
    A3: Generate 20 training data points with 2 features and assign classes
    Class 0 (Blue): Points where X + Y < 10
    Class 1 (Red): Points where X + Y >= 10
    """
    np.random.seed(42)
    X = np.random.randint(1, 11, size=(20, 2))
    # Assign classes based on sum of features (more meaningful classification)
    y = np.array([0 if (x[0] + x[1]) < 10 else 1 for x in X])
    return X, y

def classify_test_grid(X_train, y_train, k=3):
    """
    A4: Classify test grid points using kNN classifier
    Creates a fine grid of test points and classifies them
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    # Create test grid from 0 to 10 with 0.1 increments
    xx, yy = np.meshgrid(np.arange(0, 10.1, 0.1), np.arange(0, 10.1, 0.1))
    test_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(test_points)
    return xx, yy, predictions.reshape(xx.shape)

def plot_decision_boundaries(X_train, y_train, k_values):
    """
    A5: Plot decision boundaries for different k values
    Shows how class boundaries change with different k values
    """
    for k in k_values:
        xx, yy, zz = classify_test_grid(X_train, y_train, k)
        plt.figure(figsize=(6, 5))
        plt.contourf(xx, yy, zz, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolor='k')
        plt.title(f"Decision Boundary for k = {k}")
        plt.xlabel("Feature X")
        plt.ylabel("Feature Y")
        plt.grid(True)
        plt.show()

def air_quality_two_feature_classification(df, feature1, feature2, threshold_col, threshold_val):
    """
    A6: Perform two-feature classification on air quality data
    Creates binary classification based on threshold value
    """
    data = df[[feature1, feature2, threshold_col]].dropna()
    X = data[[feature1, feature2]].values
    y = (data[threshold_col] > threshold_val).astype(int).values
    return X, y

def tune_k_hyperparameter(X, y):
    """
    A7: Use GridSearchCV to find optimal k value for kNN classifier
    Performs hyperparameter tuning using cross-validation
    """
    param_grid = {'n_neighbors': list(range(1, 21))}
    model = KNeighborsClassifier()
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid.fit(X, y)
    return grid.best_params_, grid.best_score_


if __name__ == "__main__":
    print("=== Lab 3 Assignment - Classification Models ===\n")
    
    # A1: Evaluate confusion matrix and performance metrics
    print("A1: Evaluating kNN Classification Metrics...")
    try:
        air_df = pd.read_excel("AirQualityUCI.xlsx").dropna()
        X_a1 = air_df[["C6H6(GT)", "T"]].values
        y_a1 = (air_df["CO(GT)"] > 2.5).astype(int).values
        cm_train, cm_test, report_train, report_test = evaluate_knn_metrics(X_a1, y_a1, k=3)

        print("A1 - Confusion Matrix (Training):")
        print(cm_train)
        print("\nA1 - Confusion Matrix (Test):")
        print(cm_test)
        print("\nA1 - Classification Report (Training):")
        print(f"Precision: {report_train['weighted avg']['precision']:.3f}")
        print(f"Recall: {report_train['weighted avg']['recall']:.3f}")
        print(f"F1-Score: {report_train['weighted avg']['f1-score']:.3f}")
        print("\nA1 - Classification Report (Test):")
        print(f"Precision: {report_test['weighted avg']['precision']:.3f}")
        print(f"Recall: {report_test['weighted avg']['recall']:.3f}")
        print(f"F1-Score: {report_test['weighted avg']['f1-score']:.3f}")
    except Exception as e:
        print(f"Error in A1: {e}")

    # A2: Calculate regression metrics
    print("\nA2: Calculating Regression Metrics...")
    try:
        lab_df = pd.read_excel("Lab_Session_Data.xlsx")
        lab_df = lab_df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)"]].dropna()
        X_a2 = lab_df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]]
        y_a2 = lab_df["Payment (Rs)"]
        mse, rmse, mape, r2 = regression_metrics(X_a2, y_a2)

        print(f"A2 - Mean Squared Error (MSE): {mse:.2f}")
        print(f"A2 - Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"A2 - Mean Absolute Percentage Error (MAPE): {mape:.3f}")
        print(f"A2 - R² Score: {r2:.3f}")
    except Exception as e:
        print(f"Error in A2: {e}")

    # A3: Generate and plot training data
    print("\nA3: Generating Training Data...")
    X_train_a3, y_train_a3 = generate_training_data()
    plt.figure(figsize=(6, 5))
    plt.scatter(X_train_a3[:, 0], X_train_a3[:, 1], c=y_train_a3, cmap=plt.cm.coolwarm, edgecolor='k')
    plt.title("A3 - Training Data Scatter Plot (Blue: Class 0, Red: Class 1)")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.grid(True)
    plt.show()

    # A4: Classify test grid with k=3
    print("\nA4: Classifying Test Grid with k=3...")
    xx, yy, zz = classify_test_grid(X_train_a3, y_train_a3, k=3)
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X_train_a3[:, 0], X_train_a3[:, 1], c=y_train_a3, cmap=plt.cm.coolwarm, edgecolor='k')
    plt.title("A4 - Test Data Classification with k=3")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.grid(True)
    plt.show()

    # A5: Plot decision boundaries for different k values
    print("\nA5: Plotting Decision Boundaries for Different k Values...")
    plot_decision_boundaries(X_train_a3, y_train_a3, k_values=[1, 3, 5, 7, 15])

    # A6: Apply to project data (Air Quality)
    print("\nA6: Applying to Project Data (Air Quality)...")
    try:
        X_a6, y_a6 = air_quality_two_feature_classification(air_df, "PT08.S1(CO)", "T", "CO(GT)", 2.5)
        # Use first 20 points for visualization
        X_train_a6, y_train_a6 = X_a6[:20], y_a6[:20]
        plot_decision_boundaries(X_train_a6, y_train_a6, k_values=[1, 3, 5])
    except Exception as e:
        print(f"Error in A6: {e}")

    # A7: Hyperparameter tuning
    print("\nA7: Performing Hyperparameter Tuning...")
    try:
        best_k, best_score = tune_k_hyperparameter(X_a1, y_a1)
        print(f"A7 - Best k value: {best_k['n_neighbors']}")
        print(f"A7 - Best cross-validated accuracy: {best_score:.3f}")
    except Exception as e:
        print(f"Error in A7: {e}")

