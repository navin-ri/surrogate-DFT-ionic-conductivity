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
df = pd.read_csv('/src/old/20250520_two_stage/data/mpi_featurized_dataset.csv')

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

# Drop the files in the selection
clean_df = df.dropna(axis=0)

# Train-test split (80-10-10)

X = clean_df.drop(columns = ['band_gap', 'energy_above_hull', 'formation_energy_per_atom'])
y = clean_df[['band_gap', 'energy_above_hull', 'formation_energy_per_atom']]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

# To .csv
X_train.to_csv('/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/data/X_train.csv', index = False)
y_train.to_csv('/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/data/y_train.csv', index = False)
X_val.to_csv('/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/data/X_val.csv', index = False)
y_val.to_csv('/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/data/y_val.csv', index = False)
X_test.to_csv('/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/data/X_test.csv', index = False)
y_test.to_csv('/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/data/y_test.csv', index = False)