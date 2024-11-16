import joblib
import pandas as pd

# Muat model yang sudah disimpan
model = joblib.load('model/heart_disease_risk_model.joblib')

# Contoh data baru untuk prediksi
# Sesuaikan data baru ini dengan format yang sama seperti pada data pelatihan
new_data = pd.DataFrame({
    'BPXOSY1': [120, 145],  # contoh tekanan sistolik 1
    'BPXODI1': [80, 95],    # contoh tekanan diastolik 1
    'BPXOSY2': [122, 142],
    'BPXODI2': [82, 92],
    'BPXOSY3': [121, 143],
    'BPXODI3': [81, 93],
    'BPXOPLS1': [72, 78],   # contoh denyut nadi 1
    'BPXOPLS2': [70, 79],
    'BPXOPLS3': [73, 77]
})

# Gunakan model untuk melakukan prediksi
predictions = model.predict(new_data)

# Tampilkan hasil prediksi
for i, prediction in enumerate(predictions):
    risk = "High Risk" if prediction == 1 else "Low Risk"
    print(f"Sample {i+1}: {risk}")
