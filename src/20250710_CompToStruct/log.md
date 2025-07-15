# Composition features to predict the structural features

## Data preparation
dataprep.ipynb
- I remember that I used to have the space group and lattice parameters. And I already have the queried comp and struct from Material project. I can merge them together. drop only unecessary target.

    - 'src/20250710_CompToStruct/data/structural_features.csv'
    - 'src/20250710_CompToStruct/data/basic_matminer.csv'

## Model 1 
basic_model1.ipynb
- the model predicts the matminer structure features + band gap and formation energy
- one hot encode the space group, the encoder model has 'ignore unkwown' turned on, so it should work for any space group

## Data preparation for the OBELiX dataset
prep_obelix.ipynb
- Need to calculate the composition features for the formula

## important
- keep the ion char props in the base training dataset
- Need to prepare the encoder for the test data