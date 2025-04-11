import pandas as pd
import numpy as np
import random
from faker import Faker
import os

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
fake = Faker()

# Parameters
num_records = 10000
locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami', 'San Francisco', 'Seattle', 'Denver']
fraud_ratio = 0.03  # 3% fraud transactions

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Generate data
data = []
for _ in range(num_records):
    amount = round(np.random.exponential(scale=100), 2)
    location = random.choice(locations)
    timestamp = fake.date_time_between(start_date='-1y', end_date='now')
    
    is_fraud = 0
    if (amount > 500 and timestamp.hour < 6) or (random.random() < fraud_ratio):
        is_fraud = 1
    
    data.append({
        'transaction_id': fake.uuid4(),
        'amount': amount,
        'location': location,
        'timestamp': timestamp.isoformat(),
        'is_fraud': is_fraud
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("data/synthetic_transactions.csv", index=False)
print("Dataset generated and saved to data/synthetic_transactions.csv")