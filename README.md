# surrogate-DFT-ionic-conductivity

## Overview
model1 (DFT surrogate) -> model2 (ionic conductivity prediction)

## ideal plan
Prioritizing Feature Integration for Energy Above Hull

We will integrate features in the following order:

✅ High Priority (Most Impactful)
These features directly affect phase stability and should be computed first:

Bonding & Coordination
BondFractions → Describes bond types (ionic, metallic, covalent) ⏳ Medium cost
EwaldEnergy → Captures long-range electrostatic interactions ⏳ High cost
OPSiteFingerprint → Orbital-based coordination ⏳ High cost
Structural Packing & Density
DensityFeatures → Includes packing fraction, unit cell volume ⏳ Low cost
MaximumPackingEfficiency → Measures lattice energy trends ⏳ Low cost

🟡 Medium Priority (Useful but Computationally Heavy)
These features capture structural disorder and local atomic environments, which help with meta-stability but are costly: 3. Local Structure & Packing

SiteStatsFingerprint → Neighbor environment stats ⏳ Medium cost
VoronoiFingerprint → Atomic packing descriptors ⏳ Medium cost
StructuralHeterogeneity → Quantifies strain and disorder ⏳ High cost
Crystallographic Symmetry
GlobalSymmetryFeatures → Space group & symmetry ⏳ High cost

🔻 Low Priority (Limited Impact for Energy Above Hull)
These are element-based and composition-based features that are less useful for stability predictions: 5. Elemental & Composition Features

ElementProperty → Avg. electronegativity, atomic radius ⏳ Low cost
Stoichiometry → Number of elements, entropy ⏳ Low cost
IonProperty → Oxidation states, ionic character ⏳ Low cost
📌 Reason for Low Priority:

Energy above hull is structure-sensitive, and composition-based features alone cannot predict stability well.