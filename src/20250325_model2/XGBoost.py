"""
Title: XGBoost.py
Date: 2025.03.24
Description: XGBoost model for IC prediction
Version: 0.1.6
Changelog:
- 0.1.6: Cut off at 85th percentile
- 0.1.5: Cut off at 95th percentile
- 0.1.4: Raw y and PCA on X
- 0.1.3: PCA decomposition
- 0.1.2: Added inverse quantile transform
- 0.1.1: Flattened y shape
- 0.1.0: Initial version
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import QuantileTransformer
from matplotlib import pyplot as plt

# Load .csv file
X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv')

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(y, bins=50)
plt.axvline(np.percentile(y, 85), color='red', linestyle='--', label='85th percentile')
plt.title("Target Distribution with Upper Tail Cutoff")
plt.legend()
plt.show()

# Load data
X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv')

# Flatten y
y = y.to_numpy().ravel()

# Apply upper cutoff (85th percentile)
cutoff = np.percentile(y, 85)
mask = y <= cutoff

X = X[mask]
y = y[mask]

# PCA decomposition

imputer = SimpleImputer(strategy='mean')
X= imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)  # Keep 95% of variance
X = pca.fit_transform(X_scaled)

print(f"PCA reduced X from {X.shape[1]} to {X.shape[1]} components")

# Quantile transformation
#qt = QuantileTransformer(output_distribution='normal')
#y_qt = qt.fit_transform(y.to_numpy().reshape(-1, 1)).ravel()

# y log scale
y = np.log1p(y)

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

# Cross-validation: RMSE (still on log scale!)
scores = cross_val_score(
    model, X, y,
    cv=5,
    scoring='neg_mean_squared_error'
)

rmse_scores = np.sqrt(-scores)
print("RMSE (CV, log scale):", rmse_scores)
print("Average RMSE (log scale):", np.mean(rmse_scores))

# Cross-validation: R² (log scale)
r2_scores = cross_val_score(
    model, X, y,
    cv=5,
    scoring='r2'
)

print("R² scores (CV, qt scale):", r2_scores)
print("Average R² (qt scale):", np.mean(r2_scores))

# Predict using CV for parity plot
y_pred = cross_val_predict(model, X, y, cv=5)

# Inverse quantile transform
#y_pred = qt.inverse_transform(y_pred.reshape(-1, 1))

# Inverse log scale
y_pred = np.expm1(y_pred)
y_true = np.expm1(y)

# R² in original scale
r2_real = r2_score(y_true, y_pred)

from sklearn.metrics import mean_squared_error

rmse_original = np.sqrt(mean_squared_error(y, y_pred))
print("RMSE (original scale):", rmse_original)

# Plot parity
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='k')
#plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal fit')
plt.xlabel("True IC (S/cm)")
plt.ylabel("Predicted IC (S/cm)")
plt.title(f"Parity Plot (CV) — R² = {r2_real:.3f}")
#plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.scatter(np.log1p(y_true), np.log1p(y_pred), alpha=0.7, edgecolors='k')
#plt.plot([min(np.log1p(y_true)), max(np.log1p(y_true))],
#         [min(np.log1p(y_true)), max(np.log1p(y_true))], 'r--')
plt.xlabel("log(True IC)")
plt.ylabel("log(Predicted IC)")
plt.title("Log-Log Parity Plot")
plt.grid(True)
plt.tight_layout()
plt.show()