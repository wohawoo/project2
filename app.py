from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the trained model
model_path = 'speed_limit_model.h5'
model = load_model(model_path)

# Load the scaler if you have one
import pickle
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    # If scaler file doesn't exist, proceed without it
    scaler = None

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Speed limit prediction API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract features
        road_type = data.get('road_type', 0)
        traffic = data.get('traffic', 0)
        curvature = data.get('curvature', 0)
        weather = data.get('weather', 0)
        proximity_to_school = data.get('proximity_to_school', 0)
        time_of_day = data.get('time_of_day', 0)
        actual_speed = data.get('actual_speed', 0)
        
        # Prepare input for model
        input_data = np.array([[road_type, traffic, curvature, weather, 
                               proximity_to_school, time_of_day]])
        
        # If you have a scaler, apply it here
        if scaler:
            input_data = scaler.transform(input_data)
        
        # Make prediction
        predicted_speed_limit = float(model.predict(input_data)[0][0])
        
        # Prepare response
        response = {
            'predicted_speed_limit': predicted_speed_limit,
            'actual_speed': actual_speed,
            'exceeding_limit': actual_speed > predicted_speed_limit
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)