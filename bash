# Synthetic Financial Transactions Generator

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_transactions = 10000
locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami', 'San Francisco', 'Seattle']
fraud_rate = 0.02  # 2% fraud

# Generate synthetic data
def generate_transactions(n):
    data = []
    start_time = datetime(2023, 1, 1)
    
    for i in range(n):
        transaction_amount = round(np.random.exponential(scale=100), 2)  # skewed distribution
        location = random.choice(locations)
        time_offset = timedelta(minutes=np.random.randint(0, 365 * 24 * 60))
        timestamp = start_time + time_offset
        
        # Fraud probability increases with high transaction amount
        is_fraud = 1 if (random.random() < fraud_rate * (1 + (transaction_amount / 500))) else 0
        
        data.append([transaction_amount, location, timestamp, is_fraud])
    
    return pd.DataFrame(data, columns=["transaction_amount", "location", "timestamp", "fraud_flag"])

# Create dataset
df = generate_transactions(num_transactions)

# Display sample
print(df.head())

# Save to CSV
df.to_csv("synthetic_financial_transactions.csv", index=False)

# Optional: Plot some statistics
plt.figure(figsize=(10, 5))
df['transaction_amount'].hist(bins=100, edgecolor='black')
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount ($)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()