import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # To save the trained models

# Load dataset
file_path = "Data.csv"
df = pd.read_csv(file_path)

# Keep only relevant columns
df = df[['Ro', 'Su', 'Bhn', 'E']]

# Handle missing values (replace NaN with column mean)
df.fillna(df.mean(), inplace=True)

# Features (input) and target variables (output)
X = df[['Ro']]  # Mass density as input
y_su = df['Su']  # Ultimate Strength
y_bhn = df['Bhn']  # Brinell Hardness
y_e = df['E']  # Elastic Modulus

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_su_train, y_su_test = train_test_split(X_scaled, y_su, test_size=0.2, random_state=42)
X_train, X_test, y_bhn_train, y_bhn_test = train_test_split(X_scaled, y_bhn, test_size=0.2, random_state=42)
X_train, X_test, y_e_train, y_e_test = train_test_split(X_scaled, y_e, test_size=0.2, random_state=42)

# Train separate Gradient Boosting Regressor models
models = {}
for target, y_train, y_test in zip(['Su', 'Bhn', 'E'], [y_su_train, y_bhn_train, y_e_train], [y_su_test, y_bhn_test, y_e_test]):
    model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    models[target] = model

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Performance for {target}:")
    print(f"  - Mean Squared Error (MSE): {mse:.2f}")
    print(f"  - R² Score: {r2:.4f}")

    # Save the trained model
    joblib.dump(model, f"material_model_{target}.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Function to predict material properties
def predict_properties(mass_density):
    scaler = joblib.load("scaler.pkl")
    input_data = scaler.transform(np.array([[mass_density]]))

    predictions = {}
    for target in ['Su', 'Bhn', 'E']:
        model = joblib.load(f"material_model_{target}.pkl")
        predictions[target] = model.predict(input_data)[0]

    return predictions

# Take user input for mass density and predict properties
if __name__ == "__main__":
    mass_density = float(input("\nEnter mass density (Ro): "))
    predicted_values = predict_properties(mass_density)

    print("\n🔹 Predicted Properties:")
    print(f"  - Ultimate Strength (Su): {predicted_values['Su']:.2f} MPa")
    print(f"  - Brinell Hardness (Bhn): {predicted_values['Bhn']:.2f}")
    print(f"  - Elastic Modulus (E): {predicted_values['E']:.2f} MPa")
