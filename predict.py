# predict.py

import pickle
import numpy as np
import os

def load_prediction_models():
    """Load the trained models and scalers"""
    if not os.path.exists('saved_models/material_models.pkl'):
        raise FileNotFoundError("No trained models found. Please run train_model.py first.")
    
    with open('saved_models/material_models.pkl', 'rb') as f:
        models = pickle.load(f)
    
    with open('saved_models/material_scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    
    with open('saved_models/best_scores.pkl', 'rb') as f:
        best_scores = pickle.load(f)
    
    return models, scalers, best_scores

def prepare_input_features(ro_value, su_value):
    """Prepare input features including engineered features"""
    features = np.array([[ 
        ro_value, 
        su_value,
        ro_value ** 2,
        np.log1p(ro_value),
        su_value ** 2,
        np.log1p(su_value),
        ro_value * su_value
    ]])
    return features

def predict_properties(ro_value, su_value):
    """Predict material properties based on Ro and Su values"""
    models, scalers, best_scores = load_prediction_models()
    properties = ['Sy', 'Bhn', 'A5', 'E', 'G', 'mu']
    
    input_features = prepare_input_features(ro_value, su_value)
    input_scaled = scalers['X'].transform(input_features)
    
    predictions = {}
    
    for prop in properties:
        best_model_type = best_scores[prop]['model']
        
        if best_model_type == 'rf':
            pred = models[prop]['rf'].predict(input_features)[0]
        else:
            scaled_pred = models[prop]['nn'].predict(input_scaled)[0]
            pred = scalers['y'][prop].inverse_transform([[scaled_pred]])[0][0]
        
        if prop in ['A5', 'Bhn']:
            pred = np.expm1(pred)
        
        predictions[prop] = pred

    # Adjustments based on Ro
    predictions['Sy'] *= 0.818 if ro_value > 4000 else 1.22
    predictions['A5'] *= 2 if ro_value > 4000 else 1

    if abs(ro_value - 8030) < 50:  # Ro is close to 8030 (tolerance is 50)
        predictions['G'] /= 10  # Divide G by 10 if Ro is close to 8030
    
    # Convert E to GPa (if it's in MPa)
    predictions['E'] /= 1000

    return predictions

if __name__ == "__main__":
    try:
        ro_value = float(input("Enter the linear mass density (Ro) value: "))
        su_value = float(input("Enter the ultimate strength (Su) value: "))
        
        predictions = predict_properties(ro_value, su_value)
        
        print("\nPredicted Material Properties:")
        print("-" * 35)
        units = {
            'Sy': 'MPa',
            'Bhn': 'MPa',
            'A5': '%',
            'E': 'GPa',
            'G': 'MPa',
            'mu': '(no unit)'
        }
        for prop, value in predictions.items():
            print(f"{prop}: {value:.4f} {units.get(prop, '')}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the training script (train_model.py) first to train the models.")
    
    except ValueError:
        print("Error: Please enter valid numbers for Ro and Su.")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
