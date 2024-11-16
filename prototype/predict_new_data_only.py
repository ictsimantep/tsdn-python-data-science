import joblib
import pandas as pd

# Load the saved model
model = joblib.load('model/heart_disease_risk_model.joblib')

# Input 3 data points: systolic pressure, diastolic pressure, and heart rate
systolic = 120  # Example input for systolic pressure
diastolic = 80  # Example input for diastolic pressure
heart_rate = 72  # Example input for heart rate

# Construct the full feature set for prediction
# Fill the additional required features based on assumptions or placeholders
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

# Use the model to make predictions
predictions = model.predict(new_data)

# Display the prediction results
for i, prediction in enumerate(predictions):
    risk = "High Risk" if prediction == 1 else "Low Risk"
    print(f"Sample {i+1}: {risk}")
