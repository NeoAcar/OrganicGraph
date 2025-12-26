import xgboost as xgb
from dataset import MeltingPointDataset
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import torch

def main():
    parser = argparse.ArgumentParser(description='Train XGBoost Baseline')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing data')
    args = parser.parse_args()
    
    train_path = os.path.join(args.data_dir, 'train.csv')
    
    # Load Dataset
    print("Loading Data and generating fingerprints...")
    dataset = MeltingPointDataset(train_path, mode='fingerprint')
    
    X = np.array([fp for fp in dataset.fingerprints])
    y = np.array(dataset.targets)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42) # 0.1 of total (0.125 * 0.8 = 0.1)
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Train XGBoost
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=6, early_stopping_rounds=10)
    
    print("Training XGBoost...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Test RMSE: {rmse:.4f}")
    
    # Save model
    model.save_model("xgboost_model.json")

if __name__ == '__main__':
    main()
