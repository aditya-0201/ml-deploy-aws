import json
import numpy as np
import joblib

def init():
    global model
    # Load model from the deployed folder
    model = joblib.load("randomForestRegressor.pkl")

def run(data):
    try:
        # Parse JSON input
        input_data = json.loads(data)
        
        # Expecting: {"data": [[f1, f2, f3, ...]]}
        X = np.array(input_data["data"])
        
        # Make prediction
        prediction = model.predict(X)
        
        # Return result
        return {
            "prediction": prediction.tolist()
        }
    
    except Exception as e:
        return {
            "error": str(e)
        }
