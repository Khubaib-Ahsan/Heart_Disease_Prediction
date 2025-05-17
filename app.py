import pickle
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Model loading
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Model loading error: {str(e)}")
    model = None

@app.route('/')
def serve_index():
    return send_from_directory('templates', 'index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not available"}), 500
        
    try:
        data = request.get_json()
        
        # Create input DataFrame (ensure this matches your model's expected features)
        input_data = pd.DataFrame([[
            data['age'],
            data['sex'],
            data['cp'],
            data['trestbps'],
            data['chol'],
            data['fbs'],
            data['restecg'],
            data['thalch'],
            data['exang'],
            data['oldpeak'],
            data['slope'],
            data['ca'],
            data['thal']
        ]], columns=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ])
        
        prediction = int(model.predict(input_data)[0])
        
        return jsonify({
            "risk_level": prediction,
            "message": get_risk_message(prediction)[0],
            "recommendation": get_risk_message(prediction)[1]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def get_risk_message(level):
    messages = [
        ("No Significant Risk", "Continue healthy habits with regular checkups"),
        ("Mild Risk", "Monitor vitals regularly and consider lifestyle changes"),
        ("Moderate Risk", "Schedule a cardiology consultation within 2-3 months"),
        ("High Risk", "Urgent medical evaluation recommended within 2 weeks"),
        ("Critical Risk", "Seek immediate medical attention")
    ]
    return messages[level]

if __name__ == "__main__":
    app.run()
