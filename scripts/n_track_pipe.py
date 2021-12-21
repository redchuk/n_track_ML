"""
This is to join pre-processing (PCA, UMAP (LDA?)) and gradient boosting classifier in a non leaky way

I want to add PCA, UMAP (and may be LDA?) to the feature set, but retain original features; next, preprocessing is
to be combined with classifier using sklearn pipeline, so that it does not leak in CV.
FeatureUnion is, likely, way to go. Since I don't know how to retain automatically the original features,
SimpleImputer can be a workaround (I think there is no np.nan values in this data).

pipeline example
https://scikit-learn.org/stable/modules/compose.html#feature-union

Options for preprocessing:
*- ColumnTransformer - can I reuse same column several times? Seems that I can
- Can I use several Column transformers as a FeatureUnion?
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import numpy as np

data = pd.read_csv('scripts/b212a935_Chr1_data_sterile.csv')

fset_all = data.columns[6:]

fset_no_masks = [
    'f_mean_diff_xy_micron', 'f_max_diff_xy_micron', 'f_var_diff_xy_micron',
    'f_area_micron', 'f_perimeter_au_norm', 'f_min_dist_micron',
    'f_var_dist_micron', 'f_Rvar_diff_xy_micron', 'f_Rvar_dist_micron',
    'f_total_displacement', 'f_persistence',
    'f_min_dist_range', 'f_total_min_dist',
    'f_slope_min_dist_micron', 'f_slope_area_micron',
    'f_slope_perimeter_au_norm'
]
# no masks

X = data[fset_all]
y = data['t_serum_conc_percent']  # .astype('str')
y = (y / 10).astype('int')  # '10% serum' = 1, '0.3% serum' = 0

PCA_transformer = Pipeline(
    steps=[('scaler', StandardScaler()), ('PCA', PCA())]
)

c_transformer = ColumnTransformer(
    [('f_to_retain', SimpleImputer(missing_values=np.nan, strategy='mean'), fset_all),
     # (hopefully) does nothing
     ('PCA_scaled', PCA_transformer, fset_no_masks)]
)

X_t = c_transformer.fit_transform(X)
# seems it works, GET FEATURE NAMES!!!
# add classifier
