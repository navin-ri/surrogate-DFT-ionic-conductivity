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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Load .csv file
df_X = pd.read_csv('137_featurized.csv')
df_X.drop(
    columns=[
        'Original Formula',
        'MP-ID',
        'Matched Formula',
        'Structure',
        'composition',
        'Ionic Conductivity (S/cm)', # Removed IC because it has zeroes in it
    ],
    inplace=True,
)

df_X['compound possible'] = df_X['compound possible'].astype(int)

# Import ionic conductivity from .csv before .json exportation
df_y = pd.read_csv('/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250408_model2/matched_data.csv')

# Remove rows with "Not found" in the MP-ID column
df_y = df_y[df_y['MP-ID'] != "Not found"]
df_y = df_y['Ionic Conductivity (S/cm)'].astype(float)

# data split
## I/O split
X = df_X
y = df_y

print(y.describe())

## Export raw data
X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
y_l = np.log10(y)

# PCA decomposition
imputer = SimpleImputer(strategy='mean')
X= imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)
X = pca.fit_transform(X_scaled)

print(f"PCA reduced X to {X.shape[1]} components")

# Create PC column names
pca_columns = [f'PC{i+1}' for i in range(X.shape[1])]


import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.hist(y_l, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Log10 scale of Ionic conductivity (S/cm)")
plt.ylabel("Frequency")
plt.title("Distribution of IC")
plt.grid(True)
plt.tight_layout()
plt.show()

## train-val-test
X_train, X_test, y_train, y_test = train_test_split(X, y_l, test_size=0.1, random_state=42)

# Turn the datasets to dataframe with proper column names
X_train = pd.DataFrame(X_train, columns=pca_columns)
X_test = pd.DataFrame(X_test, columns=pca_columns)
y_train = pd.DataFrame(y_train.to_numpy().reshape(-1, 1), columns=['log_IC'])
y_test = pd.DataFrame(y_test.to_numpy().reshape(-1, 1), columns=['log_IC'])

# save .csv file
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)