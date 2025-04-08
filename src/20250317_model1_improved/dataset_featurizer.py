import pandas as pd
from matminer.featurizers.structure import (BondFractions,
                                            MaximumPackingEfficiency,
                                            SiteStatsFingerprint,
                                            DensityFeatures)
from matminer.featurizers.site import VoronoiFingerprint
from matminer.featurizers.composition import (ElementProperty,
                                             Stoichiometry,
                                             IonProperty)
from matminer.featurizers.conversions import StrToComposition
from pymatgen.core.structure import Structure

# import .json file
df = pd.read_json("query_data.json", orient="records")

df = df.select_dtypes(exclude=['object', 'string'])

df['structure'] = df['structure'].apply(Structure.from_dict)

# ------Initalize features----------

## Convert pretty formula to composition
stc = StrToComposition()
df = stc.featurize_dataframe(df, 'formula_pretty')

## Composition features
composition_featurizers = [
    ElementProperty.from_preset('magpie', impute_nan=True),
    Stoichiometry(),
    IonProperty(impute_nan=True),
]

## Handle BondFractions separately (needs fitting)
bond_fractions = BondFractions.from_preset('VoronoiNN')
bond_fractions.fit(df['structure'])  # Must fit

## Structure features (initialize first, skip BondFractions for now)
structure_featurizers = [
    bond_fractions,
    MaximumPackingEfficiency(),
    SiteStatsFingerprint(VoronoiFingerprint()),
    DensityFeatures(),
]

# ------Featurization----------
## Featurize compositions
for feature in composition_featurizers:
    df =  feature.featurize_dataframe(df, 'composition')

## Featurize structure
for feature in structure_featurizers:
    print(f"Featurizing with: {feature.__class__.__name__}")
    df = feature.featurize_dataframe(df, 'structure', ignore_errors=True)

# ------save to .csv----------
df.to_csv('/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250317_model1_improved/10000_featurized.csv', index = False)