from mp_api.client import MPRester
import pandas as pd

API_KEY = 'kTPLAuLmdrhseG3PJtR9ZjdF5lKNln3n'

with MPRester(API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        deprecated=False, # Don't use depreciated materials
        fields = ['material_id',
                  'formula_pretty',
                  'structure', 'band_gap',
                  'formation_energy_per_atom',
                  'energy_above_hull']
    )

# Convert the MPDataDoc into DataFrame
data = []
for doc in docs:
    data.append({
        'material_id': doc.material_id,
        'formula_pretty': doc.formula_pretty,
        'structure': doc.structure,
        'band_gap': doc.band_gap,
        'formation_energy_per_atom': doc.formation_energy_per_atom,
        'energy_above_hull': doc.energy_above_hull,
    })

# create a DataFrame
df = pd.DataFrame(data)

df['structure'] = df['structure'].apply(lambda s: s.as_dict())
df.to_json("/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250317_model1_improved/query_data.json", orient="records")
