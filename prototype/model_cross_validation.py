import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import joblib

# Path ke file .XPT
#file_path = '/Users/rioramadhan/Documents/Lomba TSDN/prototype/data/BPXO_J.XPT'
file_path = '/home/completed/tsdn/prototype/data/BPXO_J.XPT'

# 1. Muat Data
data = pd.read_sas(file_path, format='xport')

# 2. Fitur yang akan digunakan
features = ['BPXOSY1', 'BPXODI1', 'BPXOSY2', 'BPXODI2', 'BPXOSY3', 'BPXODI3', 'BPXOPLS1', 'BPXOPLS2', 'BPXOPLS3']
data_filled = data[features].fillna(data[features].median())

# 3. Buat Kolom Target 'Heart_Disease_Risk'
data_filled['Heart_Disease_Risk'] = ((data_filled[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1) >= 140) |
                                     (data_filled[['BPXODI1', 'BPXODI2', 'BPXODI3']].mean(axis=1) >= 90)).astype(int)

# 4. Pisahkan Fitur dan Target
X = data_filled[features]
y = data_filled['Heart_Disease_Risk']

# 5. Subsampling 30% dari data untuk cross-validation
X_subsample, _, y_subsample, _ = train_test_split(X, y, test_size=0.7, random_state=42)

# 6. Buat model XGBoost dengan parameter yang lebih ringan
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42, 
                          n_estimators=50, max_depth=3, subsample=0.7)

# 7. Lakukan cross-validation dengan 3 fold
cv_scores = cross_val_score(xgb_model, X_subsample, y_subsample, cv=3, scoring='accuracy')

# 8. Hitung rata-rata dan standar deviasi skor cross-validation
cv_mean_accuracy = np.mean(cv_scores)
cv_std_accuracy = np.std(cv_scores)

# Cetak hasil cross-validation
print("Cross-Validation Results:")
print(f"Mean Accuracy: {cv_mean_accuracy:.2f}")
print(f"Standard Deviation of Accuracy: {cv_std_accuracy:.2f}")

# 9. Latih Model pada Seluruh Data
xgb_model.fit(X, y)

# 10. Simpan Model yang Telah Dilatih ke File
joblib.dump(xgb_model, '/home/completed/tsdn/prototype/model/heart_disease_risk_model.joblib')
print("Model has been trained on full data and saved successfully.")
