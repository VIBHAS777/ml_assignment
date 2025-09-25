
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from lime.lime_tabular import LimeTabularExplainer

def lab_assignment_stacking_and_explanation(filepath, test_size=0.2, random_state=42, idx_to_explain=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        
        # Load data and preprocess (A1: Data preparation)
        df = pd.read_excel(filepath)
        df_clean = df.drop(['Date', 'Time'], axis=1)
        df_clean.replace(-200, np.nan, inplace=True)
        df_clean.dropna(inplace=True)
        
        # Feature-target split (target=CO(GT)) (A1)
        X = df_clean.drop('CO(GT)', axis=1)
        y = df_clean['CO(GT)']
        
        # Train-test split (A1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Define base regressors and fixed alpha Ridge meta regressor (A1)
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=random_state)),
            ('gbr', GradientBoostingRegressor(n_estimators=100, random_state=random_state))
        ]
        meta_model = Ridge(alpha=1.0)  
        
        # Construct stacking regressor (A1)
        stacking_regressor = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
        
        # Build pipeline: scaling + stacking (A2)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('stacking', stacking_regressor)
        ])
        
        # Fit pipeline on training data (A1, A2)
        pipeline.fit(X_train, y_train)
        
        # Evaluate on train and test sets (A1)
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
    
    # Fix future LIME warnings by using .iloc explicitly (A3)
    class CustomLimeTabularExplainer(LimeTabularExplainer):
        def __getitem__(self, key):
            # Override indexing to use iloc for pandas Series to avoid warnings (A3)
            if isinstance(key, int):
                return self.training_data.iloc[key]
            return super().__getitem__(key)

    # Create LIME explainer and explain single test instance (A3)
    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns.tolist(),
        mode='regression'
    )
    lime_exp = explainer.explain_instance(
        data_row=X_test.iloc[idx_to_explain],  # use iloc to avoid pandas warnings (A3)
        predict_fn=pipeline.predict
    )
    lime_exp_list = lime_exp.as_list()  # List of feature impacts (A3)

    return train_score, test_score, lime_exp_list

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    DATA_FILE = "AirQualityUCI.xlsx"
    train_r2, test_r2, lime_explanation = lab_assignment_stacking_and_explanation(DATA_FILE)
    
    print(f"Train R^2 score (A1): {train_r2:.4f}")
    print(f"Test R^2 score (A1): {test_r2:.4f}")
    print("\nLIME Explanation of test instance (A3):")
    for feature, impact in lime_explanation:
        print(f"{feature}: {impact:.4f}")
