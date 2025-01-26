from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return "Heart Disease Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Receive JSON data from frontend
        features = np.array(data['features']).reshape(1, -1)  # Convert input to NumPy array
        prediction = model.predict(features)
        result = "Positive" if prediction[0] == 1 else "Negative"
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
