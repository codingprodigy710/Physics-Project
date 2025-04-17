import pandas as pd
import joblib
from tensorflow import keras

# Load the trained model and scalers
model = keras.models.load_model("mlp_mass_density_model_v2.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Define the column names excluding 'mu' (now 6 properties)
columns = ['Su (MPa)', 'Sy (MPa)', 'A5 (%)', 'E (GPa)', 'G (MPa)', 'mu (unitless)']  # Include units

# Define the prediction function
def predict_properties(mass_density_input):
    # Scale the input
    mass_density_input_scaled = scaler_X.transform([[mass_density_input]])
    
    # Get the model's prediction
    prediction_scaled = model.predict(mass_density_input_scaled)
    
    # Inverse scale the predicted values
    prediction = scaler_y.inverse_transform(prediction_scaled)
    
    # Divide the 'mu' value by 10,000 (as requested)
    prediction[0, -1] /= 100000  # Divide the last value (mu) by 10,000
    
    # Convert the result to a DataFrame with the appropriate column names
    predicted_df = pd.DataFrame(prediction, columns=columns)
    
    return predicted_df

# User input for mass density
try:
    mass_density_input = float(input("Enter mass density (kg/m^3): "))  # Input mass density
    predicted_properties = predict_properties(mass_density_input)
    
    print(f"Predicted Properties for Mass Density {mass_density_input} kg/m^3:")
    print(predicted_properties)
except ValueError:
    print("Invalid input! Please enter a valid number for mass density.")
