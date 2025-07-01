# 

## Get_data.py
- get data via 'mp_api'
- convert the pymatgen object to ```python dict``` before json export

## test_featurize.py[100 samples]
- ```python
  # Define featurizers
  featurizers = {
    "bond_fractions": BondFractions(),
    "max_packing_efficiency": MaximumPackingEfficiency(),
    "element_properties": ElementProperty.from_preset("magpie", impute_nan=True),
    "stoichiometry": Stoichiometry(),
    "ion_properties": IonProperty(impute_nan=True),
    "density_features": DensityFeatures(),
    "site_stats_fingerprint": SiteStatsFingerprint(VoronoiFingerprint())
  }```
- elemental_poperties, and ion_properties were set to 'imipute_nan = True'
- 'ignore errors = True' was turned in structure featurization

## 10000_featurizer.py
- passed
- ~ 8 hour run time

## scikit_multi_RF.py
- decent perfomance in MultiOutputRegressor(random forest)
- passed. But energy above hull.

## IC material query
[MPQuery_for_IC.py]
- OBELiX project dataset and its formula to query the material.
- In all.csv, IC = < 1E 10 was assumed as IC = 1E 10
- 137/599 materials matched