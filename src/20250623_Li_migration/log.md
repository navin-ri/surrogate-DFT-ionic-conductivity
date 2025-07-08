# Progress log

## merging data
data_merge.ipynb 
- for merging the data from NVSE based NEB andj percolation barrier to the matminer features. Using pandas, merge method, 'left' param.
- The the NEB Li-Li hops were aggregated as mean, min, std, counts.. using pandas, agg() method.
- Note: initial trials only use BVSE method data.

## preprocessing
preprocess.ipynb
- drop_cols = ['Unnamed: 0', 'material_id', 'chemsys', '_split']
- dropna(axis =0)
- used 'loc' method to extract columns that are unique to avoid redundant constant (nunique()>1)
- train-val-test: 8:1:1 split

# train MP + BVSE model
- The predictions are worse after using the dataset maerged for BVSE. 
- Need to check how they perform against IC.

# IC preprocess
ic_preprocess.ipynb
- use the preprocessed.ipynb output (X_train.csv) as reference and picked the columns from IC dataset.

# Predict of MP+  BVSE 
neural_network.ipynb
O/P: 
    - cif_preds_only.csv
    - cif_feat_append.csv
- the trained model in the jupyternotebook was used to create the predictions of CIF feats created in ic_preprocess.ipynb

# IC prediction
ic_pred_xgb.ipynb
- The prediction accuracy not fair
- Try top features!!
- Try classification labels into it!

## Future
- The DFT NEB em have to be integrated into the merged dataset (NEB is absent :( )