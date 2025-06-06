import joblib
import numpy as np
import os
import json

# Called when the model is loaded
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    return (model, scaler)

# Called when a prediction is made
def predict_fn(input_data, model_and_scaler):
    model, scaler = model_and_scaler

    # Ensure input is a NumPy array
    if isinstance(input_data, list):
        input_data = np.array(input_data)

    # Apply scaling
    scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled)

    # Log everything to CloudWatch
    print("Input received:", json.dumps(input_data.tolist()))
    print("Scaled input:", json.dumps(scaled.tolist()))
    print("Prediction result:", json.dumps(prediction.tolist()))

    return prediction.tolist()
