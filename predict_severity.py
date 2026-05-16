import numpy as np
from model.train_model import load_model

# Load the trained model and scaler once when the script starts
try:
    MODEL, SCALER, _, _ = load_model("model/model.pkl")
except FileNotFoundError:
    print("Error: model.pkl not found. Please run train_model.py first.")
    exit(1)

def predict(temperature: float, visibility: float, humidity: float) -> float:
    """
    Predicts accident severity based on weather conditions.
    
    Args:
        temperature: Temperature in Fahrenheit
        visibility: Visibility in miles
        humidity: Humidity percentage (0-100)
        
    Returns:
        float: Predicted severity on a scale of 1.0 (minor) to 4.0 (severe)
    """
    x_norm = SCALER.transform([[temperature, visibility, humidity]])[0]
    y_norm = MODEL.predict_one(x_norm)
    return float(np.clip(y_norm * 3.0 + 1.0, 1.0, 4.0))

if __name__ == "__main__":
    # Example: Mild day, clear visibility, moderate humidity
    print(f"Predicted Severity: {predict(72.0, 10.0, 65.0):.2f}")