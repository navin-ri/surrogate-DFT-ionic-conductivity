"""
Title: validation.py
Date: 2025.03.25
Description: Evaluate log-scaled model predictions using parity plot, R², and RMSE
Version: 0.1.2
Changelog:
- 0.1.2: Added inverse log1p transform to y_true and y_pred
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# ---------------------------
# 🔧 Config: File paths
# ---------------------------
Y_TRUE_PATH = "/src/old/20250408_model2/y_test.csv"
Y_PRED_PATH = "/src/old/20250408_model2/y_pred.csv"

# ---------------------------
# 📥 Load and inverse-transform predictions
# ---------------------------
def load_data(y_true_path, y_pred_path):
    y_true_log = pd.read_csv(y_true_path).to_numpy().ravel()
    y_pred_log = pd.read_csv(y_pred_path).to_numpy().ravel()

    # Invert log → original scale
    y_true = 10 ** y_true_log
    y_pred = 10 ** y_pred_log

    return y_true, y_pred

# ---------------------------
# 📏 Evaluation metrics
# ---------------------------
def evaluate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n📊 Evaluation Results (original scale)")
    print(f"--------------------------------------")
    print(f"R² Score : {r2:.4f}")
    print(f"RMSE     : {rmse:.6f} S/cm")
    return r2, rmse

# ---------------------------
# 📈 Parity plot
# ---------------------------
# Original scale
def plot_parity(y_true, y_pred, r2):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='k')
    plt.plot([min(y_true), max(y_true)],
             [min(y_true), max(y_true)],
             'r--', label='Ideal fit')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("True IC (S/cm, log scale)")
    plt.ylabel("Predicted IC (S/cm, log scale)")
    plt.title(f"Log-log Parity Plot — R² = {r2:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------
# 🏁 Main
# ---------------------------
def main():
    y_true, y_pred = load_data(Y_TRUE_PATH, Y_PRED_PATH)
    r2, rmse = evaluate_metrics(y_true, y_pred)
    plot_parity(y_true, y_pred, r2)

# ---------------------------
# 🧪 Run if script is executed
# ---------------------------
if __name__ == "__main__":
    main()