from mp_api.client import MPRester
import pandas as pd
from matminer.featurizers.structure import (BondFractions,
                                            MaximumPackingEfficiency,
                                            SiteStatsFingerprint,
                                            DensityFeatures)
from matminer.featurizers.site import VoronoiFingerprint
from matminer.featurizers.composition import(ElementProperty,
                                             Stoichiometry,
                                             IonProperty)
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

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

# convert formula to composition
df['composition'] = df['formula_pretty'].apply(lambda x: Composition(x))

# Initialize and fit BondFractions to determine allowed bonds
def add_oxidation_states(structure):
    try:
        return structure.add_oxidation_state_by_guess()
    except Exception:
        return structure  # If it fails, return original structure

df['structure'] = df['structure'].apply(add_oxidation_states)

bf = BondFractions()
bf.fit(df['structure'].dropna().tolist())  # Fit on existing structures

# Define featurizers
featurizers = {
    "bond_fractions": bf, # use the fitted BondFractions
    "max_packing_efficiency": MaximumPackingEfficiency(),
    "element_properties": ElementProperty.from_preset("magpie", impute_nan=True),
    "stoichiometry": Stoichiometry(),
    "ion_properties": IonProperty(impute_nan=True),
    "density_features": DensityFeatures(),
    "site_stats_fingerprint": SiteStatsFingerprint(VoronoiFingerprint())
}

# Generic function to apply featurizers
def apply_featurizer(featurizer, data_column):
    def featurize(item):
        try:
            result = featurizer.featurize(item)
            return result if isinstance(result, list) else [result]
        except Exception as e:
            print(f"Error processing {data_column} with {featurizer}: {e}")
            return None

    df[f'{data_column}_features'] = df[data_column].apply(featurize)
    df_expanded = df[f'{data_column}_features'].apply(pd.Series)
    df.drop(columns=[f'{data_column}_features'], inplace=True)
    return df_expanded

# Apply featurizers
for feature_name, featurizer in featurizers.items():
    if "composition" in feature_name:
        df_expanded = apply_featurizer(featurizer, "composition")
    else:
        df_expanded = apply_featurizer(featurizer, "structure")
    df = pd.concat([df, df_expanded], axis=1)

# Display the DataFrame with all featurizers applied
print(f'The head of the DataFrame with all featurizers applied: \n {df.head()}')

df.to_csv('saved_df.csv', index = False)