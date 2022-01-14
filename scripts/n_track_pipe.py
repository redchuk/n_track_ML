"""
This is to join pre-processing (PCA, UMAP (LDA?)) and gradient boosting classifier in a non-leaky way

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
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from umap import UMAP

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

fset_raw = [
    'f_mean_diff_xy_micron', 'f_area_micron', 'f_perimeter_au_norm', 'f_min_dist_micron'
]
# raw

X = data[fset_all]
y = data['t_serum_conc_percent']  # .astype('str')
y = (y / 10).astype('int')  # '10% serum' = 1, '0.3% serum' = 0

"""
UMAP just for visualisation  
"""
"""
reducer = umap.UMAP()
scaled_all = StandardScaler().fit_transform(data[fset_all])
scaled_no_masks = StandardScaler().fit_transform(data[fset_no_masks])
scaled_raw = StandardScaler().fit_transform(data[fset_raw])

for i in [scaled_all, scaled_no_masks, scaled_raw]:
    embedding = reducer.fit_transform(i)
    sns.scatterplot(
    embedding[:, 0],
    embedding[:, 1], hue=y)
    plt.show()
    plt.close()
"""


"""
adding PCA and UMAP as features
"""

PCA_transformer = Pipeline(
    steps=[('scaler', StandardScaler()), ('PCA', PCA())]
)

UMAP_transformer = Pipeline(
    #steps=[('scaler', StandardScaler()), ('UMAP', UMAP(n_components=20))]
    steps=[('scaler', 'passthrough'), ('UMAP', UMAP(n_components=4))] # no scaling
)
c_transformer = ColumnTransformer(
    [('f_to_retain', SimpleImputer(missing_values=np.nan, strategy='mean'), fset_all),
     # (hopefully) does nothing, placed here to keep original features in pipeline output
     ('PCA_scaled', PCA_transformer, fset_no_masks),
     ('UMAP_scaled', 'passthrough', fset_raw)]
)

X_t = c_transformer.fit_transform(X)
# seems it works, is there an easy way to get feature names?

"""
PCA + UMAP + GBC pipeline
"""

boosted_forest = GradientBoostingClassifier(n_estimators=1000)
gkf = GroupKFold(n_splits=4)

gbc_pipeline = Pipeline(
    steps=[('preproc_PCA_UMAP', c_transformer),
           ('GBC', boosted_forest)]
)

b_param_grid = {'GBC__learning_rate': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
                'GBC__max_depth': [1, 2, 3, 4, 5, 10, 20, 30, 40]}

b_grid_search = GridSearchCV(gbc_pipeline, b_param_grid, cv=gkf,
                             refit=False)
b_grid_search.fit(X, y, groups=data['file'])
b_grid_forest_results = pd.DataFrame(b_grid_search.cv_results_)

b_pvt = pd.pivot_table(b_grid_forest_results,
                       values='mean_test_score',
                       index='param_GBC__learning_rate',
                       columns='param_GBC__max_depth')
sns.heatmap(b_pvt, annot=True)
plt.show()
plt.close()

"""
manual CV to get predictions out (will be emphasized in text)
best param for GBC are (max_depth 5, LR 0.1)
"""

gbc = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=5)
gkf = GroupKFold(n_splits=4)
gbc_pipeline = Pipeline(
    steps=[('preproc_PCA_UMAP', c_transformer),
           ('GBC', boosted_forest)]
)

# splitting

nuclei = data['file'].unique()
np.random.shuffle(nuclei) # does it work in place?
splits = np.array_split(nuclei, 5)

