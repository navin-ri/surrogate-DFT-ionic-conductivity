"""
Title: preprocess.py
Date: 2025.03.24
Description: preprocess 137 matched materials
Version: 0.1.0
Changelog:
- 0.1.0: Initial version
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load .csv file
df = pd.read_csv('137_featurized.csv')
df.drop(
    columns=[
        'Original Formula',
        'MP-ID',
        'Matched Formula',
        'Structure',
        'composition',
    ],
    inplace=True,
)

df['compound possible'] = df['compound possible'].astype(int)

# data split
## I/O split
X = df.drop(columns=['Ionic Conductivity (S/cm)'])
y = df['Ionic Conductivity (S/cm)']

## Export raw data
X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.hist(y, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Target (y)")
plt.ylabel("Frequency")
plt.title("Distribution of Target Variable (y)")
plt.grid(True)
plt.tight_layout()
plt.show()

y = np.log1p(df['Ionic Conductivity (S/cm)']) # Log transform IC and adds 'one', remember to us 'expm1'

## train-val-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# save .csv file
X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
y_test.to_csv('y_test.csv', index=False)