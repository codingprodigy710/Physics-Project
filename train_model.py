# Modified Training Code (train_model.py)

import pandas as pd
import numpy as np
import os
import pickle
import gc
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score
from math import log1p, expm1


class MaterialPropertiesPredictor:
    def __init__(self, data_path='Data.csv'):
        self.data_path = data_path
        self.properties = ['Sy', 'Bhn', 'A5', 'E', 'G', 'mu']  # Now predicting these, using Su as input
        self.input_features = ['Ro', 'Su']  # Using both Ro and Su as inputs
        self.models = {}
        self.scalers = {'X': StandardScaler(), 'y': {}}
        self.best_scores = {}
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.log_transform_targets = ['A5', 'Bhn']  # Apply log1p to these

        self.load_models()
    
    def clean_numeric_value(self, value):
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            match = re.search(r'(\d+\.?\d*)', str(value))
            if match:
                return float(match.group(1))
        return np.nan

    def load_data(self):
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        # Check for required columns
        required_columns = self.input_features + self.properties
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print("Cleaning numeric data...")
        for col in required_columns:
            df[col] = df[col].apply(self.clean_numeric_value)

        df = df[required_columns].dropna()
        print(f"Using {len(df)} rows after dropping missing values.")

        # Feature engineering
        df['Ro_squared'] = df['Ro'] ** 2
        df['log_Ro'] = np.log1p(df['Ro'])
        df['Su_squared'] = df['Su'] ** 2
        df['log_Su'] = np.log1p(df['Su'])
        df['Ro_Su_interaction'] = df['Ro'] * df['Su']

        # Prepare input features (X) and targets (y)
        feature_columns = ['Ro', 'Su', 'Ro_squared', 'log_Ro', 'Su_squared', 'log_Su', 'Ro_Su_interaction']
        X = df[feature_columns].values.astype(np.float32)
        y = {}

        for prop in self.properties:
            if prop in self.log_transform_targets:
                y[prop] = np.log1p(df[prop].values).astype(np.float32)
            else:
                y[prop] = df[prop].values.astype(np.float32)

        return X, y

    def split_data(self, X, y):
        X_train, X_test, _, _ = train_test_split(X, X, test_size=0.2, random_state=42)
        y_train, y_test = {}, {}
        for prop in self.properties:
            _, _, y_train[prop], y_test[prop] = train_test_split(X, y[prop], test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def preprocess_data(self, X_train, X_test, y_train):
        X_train_scaled = self.scalers['X'].fit_transform(X_train)
        X_test_scaled = self.scalers['X'].transform(X_test)

        y_train_scaled = {}
        for prop in self.properties:
            if prop not in self.scalers['y']:
                self.scalers['y'][prop] = StandardScaler()
            y_train_scaled[prop] = self.scalers['y'][prop].fit_transform(
                y_train[prop].reshape(-1, 1)).ravel()
        return X_train_scaled, X_test_scaled, y_train_scaled

    def inverse_transform_target(self, prop, values):
        unscaled = self.scalers['y'][prop].inverse_transform(values.reshape(-1, 1)).ravel()
        if prop in self.log_transform_targets:
            return np.expm1(unscaled)
        return unscaled

    def train_models(self):
        print("Starting training process...")
        try:
            X, y = self.load_data()
            if len(X) < 10:
                print("Not enough data to train.")
                return
            
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            X_train_scaled, X_test_scaled, y_train_scaled = self.preprocess_data(X_train, X_test, y_train)

            for prop in self.properties:
                print(f"\nTraining for property: {prop}")
                
                if prop not in self.models:
                    self.models[prop] = {
                        'rf': RandomForestRegressor(n_estimators=150, random_state=42, max_depth=10),
                        'nn': MLPRegressor(hidden_layer_sizes=(60, 30), max_iter=500, early_stopping=True, random_state=42)
                    }

                print("Training Random Forest...")
                self.models[prop]['rf'].fit(X_train, y_train[prop])
                rf_pred = self.models[prop]['rf'].predict(X_test)
                if prop in self.log_transform_targets:
                    rf_pred = np.expm1(rf_pred)
                    y_true = np.expm1(y_test[prop])
                else:
                    y_true = y_test[prop]
                rf_r2 = r2_score(y_true, rf_pred)
                print(f"RF R²: {rf_r2:.4f}")

                print("Training Neural Network...")
                self.models[prop]['nn'].fit(X_train_scaled, y_train_scaled[prop])
                nn_pred_scaled = self.models[prop]['nn'].predict(X_test_scaled)
                nn_pred = self.inverse_transform_target(prop, nn_pred_scaled)
                nn_r2 = r2_score(y_true, nn_pred)
                print(f"NN R²: {nn_r2:.4f}")

                best_model = 'rf' if rf_r2 >= nn_r2 else 'nn'
                best_r2 = max(rf_r2, nn_r2)

                if prop in self.best_scores:
                    if best_r2 > self.best_scores[prop]['r2']:
                        self.best_scores[prop] = {'model': best_model, 'r2': best_r2}
                        print(f"Updated best model for {prop}: {best_model} (R² = {best_r2:.4f})")
                    else:
                        print(f"Retaining previous best model for {prop}")
                else:
                    self.best_scores[prop] = {'model': best_model, 'r2': best_r2}
                    print(f"Best model for {prop}: {best_model} (R² = {best_r2:.4f})")

                self.save_models()
                gc.collect()

            print("\n✅ Training complete. Models saved.")

        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()

    def save_models(self):
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        with open('saved_models/material_models.pkl', 'wb') as f:
            pickle.dump(self.models, f)
        with open('saved_models/material_scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        with open('saved_models/best_scores.pkl', 'wb') as f:
            pickle.dump(self.best_scores, f)

    def load_models(self):
        try:
            if os.path.exists('saved_models/material_models.pkl'):
                with open('saved_models/material_models.pkl', 'rb') as f:
                    self.models = pickle.load(f)
                print("✅ Loaded previous models.")
            if os.path.exists('saved_models/material_scalers.pkl'):
                with open('saved_models/material_scalers.pkl', 'rb') as f:
                    self.scalers = pickle.load(f)
                print("✅ Loaded previous scalers.")
            if os.path.exists('saved_models/best_scores.pkl'):
                with open('saved_models/best_scores.pkl', 'rb') as f:
                    self.best_scores = pickle.load(f)
                print("✅ Loaded best scores.")
        except Exception as e:
            print(f"Error loading saved models: {e}")
            self.models, self.best_scores = {}, {}

# Main
if __name__ == "__main__":
    predictor = MaterialPropertiesPredictor()
    predictor.train_models()
