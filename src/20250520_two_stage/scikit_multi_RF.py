"""
Script name: scikit_multi_RF
description: Multi-target regression script for Random Forest
version: 0.1.0
Changelog:
- 0.1.0: Initial version
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# === Step 1: Load and merge prediction and IC data ===
cif_feat = pd.read_csv("/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/data/cif_ft.csv")
ic = pd.read_csv('/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/data/cif_ic.csv')

# Scale before PCA
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(cif_feat)

# Apply PCA to reduce to 100 components (or adjust as needed)
#pca = PCA(0.95)
#X_pca = pca.fit_transform(X_scaled)

# Save PCA-transformed features to CSV for use in NN training
#X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

# Apply log10 transformation
ic_log = np.log10(ic["Ionic Conductivity (S/cm)"])

### Quantile binning of IC for stratified split
y_bins = pd.cut(ic_log, bins=2, labels= False, duplicates= 'drop')


# === Step 3: Split and scale ===
X_train, X_val, y_train, y_val = train_test_split(cif_feat, ic_log, test_size=0.2, stratify= y_bins, random_state=42)

# Model initialization
# Model definition
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

# Evaluate metrics for each target
rmse_scores = np.sqrt(mean_squared_error(y_val, y_pred))
r2_scores = r2_score(y_val, y_pred)
# Print metrics
print(f"RMSE = {rmse_scores:.4f}, R² = {r2_scores:.4f}")

# Change to original scale
y_val = 10**y_val
y_pred = 10**y_pred

# Evaluate metrics for each target
rmse_scores = np.sqrt(mean_squared_error(y_val, y_pred))
r2_scores = r2_score(y_val, y_pred)

# Print metrics
print(f"RMSE = {rmse_scores:.4f}, R² = {r2_scores:.4f}")

# Plot parity
plt.figure(figsize=(6, 6))
plt.scatter(y_val, y_pred, alpha=0.7, edgecolors='k')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--', label='Ideal fit')
plt.xlabel("IC (S/cm)")
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Predicted IC (S/cm)")
#plt.legend()
plt.tight_layout()
plt.show()