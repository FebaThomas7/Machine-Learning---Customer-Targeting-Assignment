import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from google.colab import files
uploaded = files.upload()

# Load CSV
data = pd.read_csv("wallacecommunications.csv")

# Drop 'customer_id' if it exists
if 'customer_id' in data.columns:
    data = data.drop(columns=['customer_id'])

print(data.head())
print("Columns:", data.columns)
print("Missing values per column:\n", data.isnull().sum())
if data['new_contract_this_campaign'].dtype == object:
    data['new_contract_this_campaign'] = data['new_contract_this_campaign'].map({'yes': 1, 'no': 0})

y = data['new_contract_this_campaign']
X = data.drop(columns=['new_contract_this_campaign'])

print("Target distribution:\n", y.value_counts())
X = pd.get_dummies(X, drop_first=False)

# fill missing values
X = X.fillna(X.mean())

# Scale features
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("Sample of scaled features:\n", X_scaled.head())











