from flask import Flask, request, jsonify
import joblib
import numpy as np

# before running we need to install sckit without it python will not discover the model

app = Flask(__name__)

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
