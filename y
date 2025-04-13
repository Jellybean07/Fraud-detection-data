import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime

# Load the model
model = load_model("fraud_detection_lstm_model.h5")

# Load transaction data
df = pd.read_csv("synthetic_financial_transactions.csv")

# Feature engineering
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Encode categorical
le = LabelEncoder()
df['location'] = le.fit_transform(df['location'])

# Scale numeric features
features = ['transaction_amount', 'location', 'hour', 'day_of_week']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])
X_input = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Predict
pred_probs = model.predict(X_input)
df['fraud_probability'] = pred_probs
df['predicted_fraud'] = (df['fraud_probability'] > 0.5).astype(int)

# Flag and report
flagged = df[df['predicted_fraud'] == 1]
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
report_filename = f"fraud_report_{timestamp_str}.csv"
flagged.to_csv(report_filename, index=False)

# Download report
from google.colab import files
files.download(report_filename)

# Summary
print(f"Total Transactions: {len(df)}")
print(f"Flagged as Fraud: {len(flagged)}")