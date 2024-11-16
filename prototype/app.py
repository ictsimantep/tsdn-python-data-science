from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load the pre-trained model
model = joblib.load('model/heart_disease_risk_model.joblib')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        input_data = request.json
        systolic = float(input_data['systol'])
        diastolic = float(input_data['diastol'])
        heart_rate = float(input_data['heart_rate'])

        # Construct a DataFrame with the required 9 features
        new_data = pd.DataFrame({
            'BPXOSY1': [systolic],  # Use systolic as BPXOSY1
            'BPXODI1': [diastolic],  # Use diastolic as BPXODI1
            'BPXOSY2': [systolic + 2],  # Assume slightly higher systolic for BPXOSY2
            'BPXODI2': [diastolic + 2],  # Assume slightly higher diastolic for BPXODI2
            'BPXOSY3': [systolic + 1],  # Assume slightly higher systolic for BPXOSY3
            'BPXODI3': [diastolic + 1],  # Assume slightly higher diastolic for BPXODI3
            'BPXOPLS1': [heart_rate],  # Use heart rate as BPXOPLS1
            'BPXOPLS2': [heart_rate - 2],  # Assume slightly lower heart rate for BPXOPLS2
            'BPXOPLS3': [heart_rate + 1]  # Assume slightly higher heart rate for BPXOPLS3
        })

        # Predict using the model
        predictions = model.predict(new_data)

        # Format the output
        results = []
        for i, prediction in enumerate(predictions):
            risk = "High Risk" if prediction == 1 else "Low Risk"
            results.append({
                "sample": i + 1,
                "risk": risk
            })

        # Return formatted JSON response
        return jsonify(results)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
