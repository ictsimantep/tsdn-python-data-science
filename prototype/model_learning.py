import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Path ke file .XPT
#file_path = '/Users/rioramadhan/Documents/Lomba TSDN/prototype/data/BPXO_L.XPT'
file_path = '/home/completed/tsdn/prototype/data/BPXO_L.XPT'

# 1. Muat Data
data = pd.read_sas(file_path, format='xport')

# 2. Fitur yang akan digunakan
features = ['BPXOSY1', 'BPXODI1', 'BPXOSY2', 'BPXODI2', 'BPXOSY3', 'BPXODI3', 'BPXOPLS1', 'BPXOPLS2', 'BPXOPLS3']

# 3. Isi Nilai yang Hilang dengan Median
data_filled = data[features].fillna(data[features].median())

# 4. Buat Kolom Target 'Heart_Disease_Risk'
# Misalnya, tekanan darah sistolik rata-rata >= 140 atau diastolik >= 90 dianggap berisiko tinggi
data_filled['Heart_Disease_Risk'] = ((data_filled[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1) >= 140) |
                                     (data_filled[['BPXODI1', 'BPXODI2', 'BPXODI3']].mean(axis=1) >= 90)).astype(int)

# 5. Pisahkan Fitur dan Target
X = data_filled[features]
y = data_filled['Heart_Disease_Risk']

# 6. Split Data menjadi Training dan Testing (80-20 Split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model XGBoost tanpa early stopping
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)  # Hanya gunakan fit() biasa tanpa eval_set atau early_stopping_rounds
xgb_pred = xgb_model.predict(X_test)

# 8. Evaluasi Model
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred)
xgb_recall = recall_score(y_test, xgb_pred)

# 9. Tampilkan Hasil
print("XGBoost Results:")
print(f"Accuracy: {xgb_accuracy:.2f}")
print(f"Precision: {xgb_precision:.2f}")
print(f"Recall: {xgb_recall:.2f}")
