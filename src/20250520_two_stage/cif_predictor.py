import pandas as pd
import torch
import torch.nn as nn
import joblib
import numpy as np

# === 1. Load featurized input data ===
input_path = "/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/data/cif_ft.csv"  # Update path if needed
df = pd.read_csv(input_path)

# === 2. Load scaler and transform features ===
scaler = joblib.load("/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/saved_states/scaler.pkl")
X_scaled = scaler.transform(df)

# === 3. Define the model architecture (must match training) ===
class Architecture(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

# === 4. Load the trained model ===
model = Architecture(input_dim=X_scaled.shape[1])
model.load_state_dict(torch.load("/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/saved_states/best_surrogate_model.pt"))
model.eval()

# === 5. Predict using the model ===
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
with torch.no_grad():
    y_pred_tensor = model(X_tensor)

y_pred_np = y_pred_tensor.numpy()

# === 6. Add predictions back to original DataFrame ===
predictions_df = pd.DataFrame(
    y_pred_np,
    columns=["Predicted band_gap", "Predicted energy_above_hull", "Predicted formation_energy_per_atom"],
    index=df.index
)

result_df = pd.concat([df, predictions_df], axis=1)

# === 7. Save final prediction dataset ===
output_path = "/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/data/cif_predictions.csv"
result_df.to_csv(output_path, index=False)
print(f"✅ Predictions saved to {output_path}")