import os

import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure as PmgStructure
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
from matminer.featurizers.structure import DensityFeatures
from matminer.featurizers.composition import ElementProperty
import warnings

warnings.simplefilter('ignore')

from pymatgen.io.cif import CifParser
from pymatgen.core import Structure, Species

def round_structure_hard(structure, threshold=0.5):
    """
    Hard round occupancies at each site.
    Keep ONLY the first species with occupancy >= threshold.
    Delete all others.
    """
    new_species = []
    new_coords = []

    for site in structure.sites:
        new_site_species = {}

        # Keep only the FIRST species with occupancy >= threshold
        for sp, occu in site.species.items():
            if occu >= threshold:
                new_site_species[sp] = 1.0  # Keep this one species
                break  # Stop after first

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

# Get CIF files in list
cif_directory = '/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250428_CIF_test/cifs'
cif_list = [f for f in os.listdir(cif_directory) if f.endswith('.cif')]

# Define the featurizers
struct_feats = DensityFeatures()
comp_feats = ElementProperty.from_preset('magpie')

# Get feature names first
struct_labels = struct_feats.feature_labels()
comp_labels = comp_feats.feature_labels()
all_labels = comp_labels + struct_labels + ["cif_filename"]

# Empty list for collected features
all_data = []

# Loop
for cif in cif_list:
    try:
        cif_path = os.path.join(cif_directory, cif)

        # Load CIF using tolerant parser
        parser = CifParser(cif_path, occupancy_tolerance=2.0)
        structures = parser.get_structures(primitive=False)
        structure = structures[0]

        # Round the occupancies
        rounded_structure = round_structure_hard(structure)

        # Featurize structure
        density = struct_feats.featurize(rounded_structure)
        if not isinstance(density, list):
            density = [density]  # Make sure it's a list

        # Featurize composition
        element = comp_feats.featurize(rounded_structure.composition)
        if not isinstance(element, list):
            element = [element]  # Make sure it's a list

        # Combine features
        feats = element + density
        feats.append(cif)  # Add filename

        # Add to list
        all_data.append(feats)

    except Exception as e:
        print(f'Error processing {cif}: {e}')

# Convert to DataFrame
features_df = pd.DataFrame(all_data, columns=all_labels)

# Save
features_df.to_csv(
    '/Users/navin/Library/CloudStorage/Dropbox-AIZOTH/研究/Navin/NIMS/surrogate-DFT-ionic-conductivity/src/20250428_CIF_test/cif_features.csv',
    index=False)