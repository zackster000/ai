import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import sys

# Add the project root to the path so we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocess import preprocess_data

def train():
    # Load dataset with absolute path relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'consumption.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found.")
        return
    
    df = pd.read_csv(data_path)
    
    # Preprocess
    df_processed = preprocess_data(df)
    
    # Features and target
    X = df_processed[['day_of_week', 'month', 'is_weekend', 'lag_1']]
    y = df_processed['consumption']
    
    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # 1. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_r2 = r2_score(y_test, lr_preds)
    
    # 2. Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)
    
    # Comparison Results
    print("-" * 30)
    print("Model Evaluation Results:")
    print("-" * 30)
    print(f"Linear Regression: MAE = {lr_mae:.4f}, R2 = {lr_r2:.4f}")
    print(f"Random Forest:     MAE = {rf_mae:.4f}, R2 = {rf_r2:.4f}")
    print("-" * 30)
    
    # Choose best model (by MAE)
    if rf_mae < lr_mae:
        best_model = rf_model
        best_name = "RandomForestRegressor"
        metrics = {"mae": rf_mae, "r2": rf_r2}
    else:
        best_model = lr_model
        best_name = "LinearRegression"
        metrics = {"mae": lr_mae, "r2": lr_r2}
        
    print(f"Saving BEST model: {best_name}")
    
    # Save the best model and its metrics with absolute path
    save_path = os.path.join(base_dir, 'saved_model.pkl')
    model_data = {
        'model': best_model,
        'model_name': best_name,
        'metrics': metrics,
        'feature_names': list(X.columns)
    }
    joblib.dump(model_data, save_path)
    print(f"Model saved as {save_path}")

if __name__ == "__main__":
    train()
