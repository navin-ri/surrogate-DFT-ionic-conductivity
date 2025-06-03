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
df = pd.read_csv('/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/data/cif_features_full.csv')

# Dropping the unnecessary columns
df = df.drop(
    columns=[
        'formula_pretty',
        'ID',
        'Structure',
        'composition',
        ], axis = 1
    )

# Convert the 'compound possible' column to binary input
df['compound possible'] = df['compound possible'].astype(int)

# Drop the files in the selection
clean_df = df.dropna(axis=0)

X = clean_df.drop(columns = ['Ionic Conductivity (S/cm)'])
y = clean_df['Ionic Conductivity (S/cm)']

# To .csv
X.to_csv('/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/data/cif_ft.csv', index = False)
y.to_csv('/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/data/cif_ic.csv', index = False)