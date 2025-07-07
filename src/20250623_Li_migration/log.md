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

# predict MP + BVSE properties
- The predictions are worse after using the dataset maerged for BVSE. 
- Need to check how they perform against IC.

# IC preprocess
ic_preprocess.ipynb
- use the preprocessed.ipynb output (basic_matminer.csv) as reference and picked the columns from IC dataset.

## Future
- The DFT NEB em have to be integrated into the merged dataset