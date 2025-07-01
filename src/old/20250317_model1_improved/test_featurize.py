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
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

# import the
df = pd.read_json("/src/old/20250317_model1_improved/query_data.json", orient="records")
df['structure'] = df['structure'].apply(Structure.from_dict)

# Test with a small subset of the dataset: 10000
df = df[1000:1100]

# Convert pretty formula to composition
stc = StrToComposition()
df = stc.featurize_dataframe(df, 'formula_pretty')

## Featurize composition
### Elemental property
ep = ElementProperty.from_preset('magpie', impute_nan=True)
df = ep.featurize_dataframe(df, 'composition')

### Stochiometry
st = Stoichiometry()
df = st.featurize_dataframe(df, 'composition')

### Ion property
ip = IonProperty(impute_nan= True)
df = ip.featurize_dataframe(df, 'composition')

# Featurize structure
### Bond fraction
ip = BondFractions.from_preset('VoronoiNN')
ip.fit(df['structure'])
df = ip.featurize_dataframe(df, 'structure')

### Maximum packing efficiency
mpe = MaximumPackingEfficiency()
df = mpe.featurize_dataframe(df, 'structure')

### Site stat fingerprint
ssf = SiteStatsFingerprint(VoronoiFingerprint())
df = ssf.featurize_dataframe(df, 'structure')

# Save to .csv
df.to_csv('/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250317_model1_improved/test_featurized.csv', index = False)