import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime

# Load model
model = load_model("fraud_detection_lstm_model.h5")  # Save your model with this filename

# Load new transactions
df = pd.read_csv("synthetic_financial_transactions.csv")

# Preprocessing (same steps used in training)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

le = LabelEncoder()
df['location'] = le.fit_transform(df['location'])

features = ['transaction_amount', 'location', 'hour', 'day_of_week']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])
X_input = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Predict fraud probability
pred_probs = model.predict(X_input)
df['fraud_probability'] = pred_probs
df['predicted_fraud'] = (df['fraud_probability'] > 0.5).astype(int)

# Flag suspicious transactions
flagged = df[df['predicted_fraud'] == 1]

# Generate summary report
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
report_filename = f"fraud_report_{timestamp_str}.csv"
flagged.to_csv(report_filename, index=False)

print(f"[INFO] Total Transactions: {len(df)}")
print(f"[INFO] Suspicious Transactions Detected: {len(flagged)}")
print(f"[INFO] Report saved as: {report_filename}")