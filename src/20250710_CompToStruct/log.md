# Composition features to predict the structural features

## data preparation
dataprep.ipynb
- I remember that I used to have the space group and lattice parameters. And I already have the queried comp and struct from Material project. I can merge them together.
    - 'src/20250710_CompToStruct/data/structural_features.csv'
    - 'src/20250710_CompToStruct/data/basic_matminer.csv'

## Model 1 
basic_model1.ipynb
- the model predicts the matminer structure features + band gap and formation energy
- one hot encode the crystal system and space groups