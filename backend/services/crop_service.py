"""
Crop Recommendation Service
============================
This service loads the trained crop recommendation model and provides
a clean interface for predicting the best crop to grow based on
soil and weather conditions.

Part of the "Agentic AI Assistant for Sustainable Farming Decisions" project.
Week 2: Crop Recommendation Module
"""

import joblib
import pandas as pd
import os


class CropRecommendationService:
    """
    A service class that handles crop recommendation predictions.
    
    This class loads the trained ML model once (on initialization)
    and provides a method to predict crops for given farming conditions.
    """

    def __init__(self):
        """Load the trained model, scaler, and label encoder."""
        # Get the directory where this file is located
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        model_path = os.path.join(base_dir, 'models', 'crop_model.pkl')
        scaler_path = os.path.join(base_dir, 'scalers', 'crop_scaler.pkl')
        encoder_path = os.path.join(base_dir, 'encoders', 'label_encoder.pkl')
        
        # Load the model
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"✅ Crop model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the scaler (optional - used for Logistic Regression)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = None
            print("⚠️ Scaler not found. Predictions will use unscaled data.")
        
        # Load the label encoder (optional - used for XGBoost)
        if os.path.exists(encoder_path):
            self.label_encoder = joblib.load(encoder_path)
        else:
            self.label_encoder = None
            print("⚠️ Label encoder not found.")

        # Define valid ranges for each input
        self.valid_ranges = {
            'nitrogen':    (0, 300, 'kg/ha'),
            'phosphorus':  (0, 300, 'kg/ha'),
            'potassium':   (0, 300, 'kg/ha'),
            'temperature': (-10, 60, '°C'),
            'humidity':    (0, 100, '%'),
            'ph':          (0, 14, 'pH scale'),
            'rainfall':    (0, 5000, 'mm')
        }

    def validate_inputs(self, nitrogen, phosphorus, potassium,
                        temperature, humidity, ph, rainfall):
        """
        Validate all input parameters.
        
        Returns:
            tuple: (is_valid: bool, error_message: str or None)
        """
        # Check that all inputs are numeric
        inputs = {
            'nitrogen': nitrogen,
            'phosphorus': phosphorus,
            'potassium': potassium,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }
        
        for name, value in inputs.items():
            # Check if value is numeric
            if not isinstance(value, (int, float)):
                return False, f"'{name}' must be a number. Got {type(value).__name__}."
            
            # Check if value is finite (not NaN or Inf)
            if not pd.notna(value) or not pd.api.types.is_number(value):
                return False, f"'{name}' must be a finite number. Got {value}."
            
            # Check if value is within valid range
            min_val, max_val, unit = self.valid_ranges[name]
            if value < min_val or value > max_val:
                return False, (
                    f"'{name}' must be between {min_val} and {max_val} {unit}. "
                    f"Got {value}."
                )
        
        return True, None

    def predict(self, nitrogen, phosphorus, potassium,
                temperature, humidity, ph, rainfall):
        """
        Predict the recommended crop based on soil and weather conditions.
        
        Parameters:
            nitrogen (float):    Nitrogen content in soil (kg/ha)
            phosphorus (float):  Phosphorus content in soil (kg/ha)
            potassium (float):   Potassium content in soil (kg/ha)
            temperature (float): Average temperature (°C)
            humidity (float):    Average relative humidity (%)
            ph (float):          Soil pH value
            rainfall (float):    Average rainfall (mm)
        
        Returns:
            dict: {
                "recommended_crop": str,
                "confidence": float or None,
                "input_summary": dict
            }
            or
            dict: {"error": str} if validation fails
        """
        # Step 1: Validate inputs
        is_valid, error_msg = self.validate_inputs(
            nitrogen, phosphorus, potassium,
            temperature, humidity, ph, rainfall
        )
        
        if not is_valid:
            return {"error": error_msg}
        
        # Step 2: Create input DataFrame (must match training column names)
        input_data = pd.DataFrame([{
            'N': nitrogen,
            'P': phosphorus,
            'K': potassium,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }])
        
        # Step 3: Make prediction
        prediction = self.model.predict(input_data)[0]
        
        # Step 4: Get prediction confidence (probability) if available
        confidence = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(input_data)[0]
            confidence = float(max(probabilities))
        
        # Step 5: Return structured result
        result = {
            "recommended_crop": str(prediction),
            "confidence": round(confidence, 4) if confidence is not None else None,
            "input_summary": {
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall
            }
        }
        
        return result


# ============================================================
# Convenience function for simple usage without creating a class
# ============================================================
_service_instance = None

def get_crop_service():
    """Get or create the singleton service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = CropRecommendationService()
    return _service_instance


def predict_crop(nitrogen, phosphorus, potassium,
                 temperature, humidity, ph, rainfall):
    """
    Simple function to predict a crop.
    
    Usage:
        result = predict_crop(90, 42, 43, 25.5, 80, 6.5, 200)
        print(result)
        # {'recommended_crop': 'rice', 'confidence': 0.97, ...}
    """
    service = get_crop_service()
    return service.predict(
        nitrogen, phosphorus, potassium,
        temperature, humidity, ph, rainfall
    )