import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

# Model configuration - must match exactly what the model expects
EXPECTED_FEATURES = [
    'fbs', 'sex', 'restecg', 'dataset', 'cp', 'thal', 'exang', 'slope',
    'trestbps', 'chol', 'thalch', 'age', 'oldpeak', 'ca'
]

try:
    with open("Heart_Disease_Prediction/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Model loading error: {str(e)}")
    model = None

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not available"}), 500
        
    try:
        data = request.get_json()
        
        # Create DataFrame with columns in exact order the model expects
        input_data = pd.DataFrame([[
            data['fbs'],
            data['sex'],
            data['restecg'],
            data['dataset'],
            data['cp'],
            data['thal'],
            data['exang'],
            data['slope'],
            data['trestbps'],
            data['chol'],
            data['thalch'],
            data['age'],
            data['oldpeak'],
            data['ca']
        ]], columns=EXPECTED_FEATURES)
        
        prediction = int(model.predict(input_data)[0])
        
        return jsonify({
            "risk_level": prediction,
            "message": get_risk_message(prediction)[0],
            "recommendation": get_risk_message(prediction)[1],
            "disclaimer": "Results are predictions based on statistical analysis and should not replace professional medical advice."
        })
        
    except KeyError as e:
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
    app.run(host="0.0.0.0", port=5000)
