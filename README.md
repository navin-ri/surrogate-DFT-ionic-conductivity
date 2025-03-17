# Ionic conductivity prediction from structural parameters

## Brain storm
### 2024.02.11
1. Use persistent diagrams to build a CNN model
2. MP -> I/P -> Model 1 -> O/P -> m1_prop

   |---------------PD-----------> PD feat.

    m1_prop + PD feat-> I/P -> model 2 -> IC 

3. other option is to use matminer to  featurizr all inputd and elec properties from matproject and predict IC finally. ?
   - It is computationally expensive to featurize RDF, etc.
   - Needs dim. red. to reduce overfitting

### Index (trial/...)
1. 20250221_model_1_pca_basic: 