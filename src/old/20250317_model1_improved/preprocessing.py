"""
Script name: preprocessing
description: Preprocessing script for neural network model
version: 0.1.0
Changelog:
- 0.1.0: Initial version
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Import the dataset
df = pd.read_csv('10000_featurized.csv')

# Dropping the unnecessary columns
df = df.drop(
    columns=[
        'formula_pretty',
        'material_id',
        'structure',
        'composition',
        ], axis = 1
    )

# Convert the 'compound possible' column to binary input
df['compound possible'] = df['compound possible'].astype(int)

# Train-test split (80-10-10)

X = df.drop(columns = ['band_gap', 'energy_above_hull', 'formation_energy_per_atom'])
y = df[['band_gap', 'energy_above_hull', 'formation_energy_per_atom']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# To .csv
X_train.to_csv('X_train.csv', index = False)
y_train.to_csv('y_train.csv', index = False)
X_val.to_csv('X_val.csv', index = False)
y_val.to_csv('y_val.csv', index = False)
X_test.to_csv('X_test.csv', index = False)
y_test.to_csv('y_test.csv', index = False)
