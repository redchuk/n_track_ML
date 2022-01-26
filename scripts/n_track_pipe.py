"""
This is to join pre-processing (PCA, UMAP) and gradient boosting classifier in a non-leaky way

I want to add PCA and UMAP to the feature set, but retain original features; next, preprocessing is
to be combined with classifier using sklearn pipeline, so that it does not leak in CV.

"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_score, cross_val_predict
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
    # steps=[('scaler', StandardScaler()), ('UMAP', UMAP(n_components=20))]
    steps=[('scaler', 'passthrough'), ('UMAP', UMAP(n_components=4))]  # no scaling
)
c_transformer = ColumnTransformer(
    [('f_to_retain', SimpleImputer(missing_values=np.nan, strategy='mean'), fset_all),
     # (hopefully) does nothing, placed here to keep original features in pre-processing pipeline output
     ('PCA_scaled', PCA_transformer, fset_no_masks),
     ('UMAP_scaled', 'passthrough', fset_raw)]
)

# X_t = c_transformer.fit_transform(X)
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
manual CV to get predictions out (will be explained in text)
best param for GBC are (max_depth 5, LR 0.1(should be default))
"""

gbc = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=5)
gbc_pipeline = Pipeline(
    steps=[('preproc_PCA_UMAP', c_transformer),
           ('GBC', gbc)]
)

# splitting (instead of gkf=4 above)
nuclei = data['file'].unique()
np.random.shuffle(nuclei)
splits = np.array_split(nuclei, 4)

scores = []
print('manual splits:')
# for split in splits:
for strain,stest in gkf.split(X,y,data['file']):
    # test_data = data[data['file'].isin(split)]
    # train_data = data[~data['file'].isin(split)]
    train_data = data.iloc[strain,:]
    test_data = data.iloc[stest,:]
    X = train_data[fset_all]
    y = train_data['t_serum_conc_percent']
    y = (y / 10).astype('int')
    X_test = test_data[fset_all]
    y_test = test_data['t_serum_conc_percent']
    y_test = (y_test / 10).astype('int')

    gbc_pipeline.fit(X, y)
    print(X.shape, y.shape, X_test.shape, y_test.shape)
    print(gbc_pipeline.score(X_test, y_test))
    scores.append(gbc_pipeline.score(X_test, y_test))
print(scores)
print(np.mean(scores))

"""
automated splits with fixed GBC hyperparameters
"""

X = data[fset_all]
y = data['t_serum_conc_percent']  # .astype('str')
y = (y / 10).astype('int')  # '10% serum' = 1, '0.3% serum' = 0

gbc = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=5)
gkf = GroupKFold(n_splits=4)
gbc_pipeline = Pipeline(
    steps=[('preproc_PCA_UMAP', c_transformer),
           ('GBC', gbc)]
)

scores = cross_val_score(gbc_pipeline, X, y, cv=gkf, groups=data['file'])
print('automated splits, same hyperparameters:')
print(scores)
print(np.mean(scores))

cv_pred = cross_val_predict(gbc_pipeline, X, y, cv=gkf, groups=data['file'])  # returns predictions from cv
(y == cv_pred).sum() / len(y)  # accuracy for cv-derived predictions
#  cross_val_predict is not an appropriate measure of generalisation error, see docs




