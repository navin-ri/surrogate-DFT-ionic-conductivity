import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import numpy as np

# === Step 1: Load and merge prediction and IC data ===
cif_feat = pd.read_csv("/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/data/cif_predictions.csv")
ic = pd.read_csv('/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/data/cif_ic.csv')

# Apply log10 transformation
ic_log = np.log10(ic["Ionic Conductivity (S/cm)"])

### Quantile binning of IC for stratified split
y_bins = pd.cut(ic_log, bins=3, labels= False, duplicates= 'drop')


# === Step 3: Split and scale ===
X_train, X_val, y_train, y_val = train_test_split(cif_feat, ic_log, test_size=0.2, stratify= y_bins, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "ic_scaler.pkl")

# === Step 4: Convert to PyTorch tensors ===
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32)

# === Step 5: Define model ===
class ICModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = ICModel(input_dim=cif_feat.shape[1])
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === Step 6: Training loop ===
best_val_loss = float("inf")

for epoch in range(50):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor)
        val_loss = loss_fn(val_pred, y_val_tensor)

    print(f"Epoch {epoch:03} - Train Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        torch.save(model.state_dict(), "/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/saved_states/best_ic_model.pt")

# === Step 7: Evaluation ===
model.load_state_dict(torch.load("/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/saved_states/best_ic_model.pt"))
model.eval()

with torch.no_grad():
    test_pred = model(X_test_tensor).cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()

    # Log-scale RMSE & R²
    rmse_log = mean_squared_error(y_test_np, test_pred, squared=False)
    r2_log = r2_score(y_test_np, test_pred)

    # Back-transform to original scale
    test_pred_orig = np.power(10, test_pred)
    y_test_orig = np.power(10, y_test_np)
    rmse_orig = mean_squared_error(y_test_orig, test_pred_orig, squared=False)
    r2_orig = r2_score(y_test_orig, test_pred_orig)

    print(f"\n✅ Log-scale Test RMSE: {rmse_log:.4f}")
    print(f"✅ Log-scale Test R²: {r2_log:.4f}")
    print(f"✅ Original-scale Test RMSE: {rmse_orig:.4e}")
    print(f"✅ Original-scale Test R²: {r2_orig:.4f}")

# === Step 8: Log-Scale Parity Plot ===
plt.figure(figsize=(6, 6))
plt.scatter(y_test_np, test_pred, alpha=0.5)
plt.plot([y_test_np.min(), y_test_np.max()],
         [y_test_np.min(), y_test_np.max()], 'k--')
plt.xlabel("True log₁₀(IC)")
plt.ylabel("Predicted log₁₀(IC)")
plt.title("Parity Plot: Predicted vs True log₁₀(IC)")
plt.tight_layout()
#plt.savefig("ic_log_parity_plot.png", dpi=300)
plt.show()
