import pandas as pd
import numpy as np

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
