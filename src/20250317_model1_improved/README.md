# 

## Get_data.py
- get data via 'mp_api'

- `# Define featurizers
  featurizers = {
    "bond_fractions": BondFractions(),
    "max_packing_efficiency": MaximumPackingEfficiency(),
    "element_properties": ElementProperty.from_preset("magpie", impute_nan=True),
    "stoichiometry": Stoichiometry(),
    "ion_properties": IonProperty(impute_nan=True),
    "density_features": DensityFeatures(),
    "site_stats_fingerprint": SiteStatsFingerprint(VoronoiFingerprint())
  }`

- elemental_poperties, and ion_properties were set to 'imipute_nan = True'