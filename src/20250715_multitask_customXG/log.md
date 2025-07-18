# Composition features to predict the structural features

## Data preparation
dataprep.ipynb
- convert space group to space number using pymatgen
- Merge the ED and Em samples
    - 'src/20250710_CompToStruct/data/structural_features.csv'
    - 'src/20250710_CompToStruct/data/basic_matminer.csv'

## Model 1
basic_model2.ipynb (cell extension of basic_model1.ipynb)
- the trained model using model 1 to featurize the obelix dataset for structural desc.
- Multi-task NN was used to separate prediction labels based on the properties, thermodynamic, topological, ...
- The EM and EDs were masked in the same model along with multi-task NN.

## Data preparation for the OBELiX dataset
prep_obelix.ipynb
- Need to calculate the composition features for the formula
- remove space group and use the space group number (pymatgen)