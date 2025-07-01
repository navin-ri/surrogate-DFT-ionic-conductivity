import os
import pandas as pd
import joblib
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure
from matminer.featurizers.structure import (
    BondFractions,
    MaximumPackingEfficiency,
    SiteStatsFingerprint,
    DensityFeatures
)
from matminer.featurizers.site import VoronoiFingerprint
from matminer.featurizers.composition import (
    ElementProperty,
    Stoichiometry,
    IonProperty
)
from matminer.featurizers.conversions import StrToComposition
import warnings

warnings.simplefilter('ignore')

def main():
    def round_structure_hard(structure, threshold=0.5):
        new_species = []
        new_coords = []
        for site in structure.sites:
            new_site_species = {}
            for sp, occu in site.species.items():
                if occu >= threshold:
                    new_site_species[sp] = 1.0
                    break
            if new_site_species:
                new_species.append(new_site_species)
                new_coords.append(site.frac_coords)
        if not new_species:
            raise ValueError("All species removed after hard rounding.")
        return Structure(
            lattice=structure.lattice,
            species=new_species,
            coords=new_coords,
            to_unit_cell=True
        )

    cif_directory = '/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/data/cifs'
    cif_list = [f for f in os.listdir(cif_directory) if f.endswith('.cif')]

    structure_data = []
    for cif in cif_list:
        try:
            cif_path = os.path.join(cif_directory, cif)
            parser = CifParser(cif_path, occupancy_tolerance=2.0)
            structure = parser.get_structures(primitive=False)[0]
            rounded_structure = round_structure_hard(structure)
            structure_data.append({
                "ID": cif.replace('.cif',''),
                "formula_pretty": rounded_structure.composition.reduced_formula,
                "Structure": rounded_structure,
                "composition": rounded_structure.composition
            })
        except Exception as e:
            print(f"Error processing {cif}: {e}")

    df = pd.DataFrame(structure_data)
    print(f"✅ Successfully loaded {len(df)} structures.")

    # --- Step 2: Composition features
    composition_featurizers = [
        ElementProperty.from_preset('magpie', impute_nan=True),
        Stoichiometry(),
        IonProperty(impute_nan=True),
    ]
    for feature in composition_featurizers:
        print(f"Featurizing compositions with: {feature.__class__.__name__}")
        df = feature.featurize_dataframe(df, 'composition')

    # --- Step 3: Load fitted BondFractions
    bond_fractions = joblib.load("/src/old/20250520_two_stage/saved_states/bond_fractions_fitted.pkl")

    # --- Step 4: Structure features
    structure_featurizers = [
        bond_fractions,
        MaximumPackingEfficiency(),
        SiteStatsFingerprint(VoronoiFingerprint()),
        DensityFeatures(),
    ]
    for feature in structure_featurizers:
        print(f"Featurizing structures with: {feature.__class__.__name__}")
        df = feature.featurize_dataframe(df, 'Structure', ignore_errors=True)

    # --- Step 5: Add ionic conductivity values
    raw = pd.read_csv('/src/old/20250520_two_stage/data/raw.csv', dtype={'ID': str})

    def get_ic(cif_id):
        match = raw[raw['ID'] == cif_id]
        if not match.empty:
            return match['Ionic conductivity (S cm-1)'].values[0]
        else:
            return None

    df['Ionic Conductivity (S/cm)'] = df['ID'].apply(get_ic)

    # --- Step 6: Save result
    df.to_csv(
        '/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250520_two_stage/data/cif_features_full.csv',
        index=False)
    print("✅ Full featurization complete. Saved to cif_features_full.csv")

if __name__ == "__main__":
    main()
