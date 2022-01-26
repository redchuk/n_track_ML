import pandas as pd
import numpy as np
import shap
from scipy.stats import linregress
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import seaborn as sns
from matplotlib import pyplot as plt

''' 
read the data 
'''

data = pd.read_csv('scripts/a286935_data_chromatin_live.csv')
data = data[~data["comment"].isin(["stress_control"])]
data = data[~data["comment"].isin(["H2B"])]
data = data[data["guide"].str.contains('1398') | data["guide"].str.contains('1514')]
data = data[data["time"] < 40]

# initial filtering based on experimental setup

''' 
add features 

In a resulting table target column names start with a 't', while features to be used in training start with 'f'.
'''

data_agg = data.groupby(['file', 'particle']).agg(t_guide=('guide', 'first'),
                                                  t_time=('time', 'first'),
                                                  t_serum_conc_percent=('serum_conc_percent', 'first'),

                                                  f_mean_diff_xy_micron=('diff_xy_micron', 'mean'),
                                                  # average displacement
                                                  f_max_diff_xy_micron=('diff_xy_micron', 'max'),
                                                  # maximal displacement
                                                  f_sum_diff_xy_micron=('diff_xy_micron', 'sum'),
                                                  # total trajectory length
                                                  f_var_diff_xy_micron=('diff_xy_micron', 'var'),
                                                  # variance in displacements

                                                  sum_diff_x_micron=('diff_x_micron', 'sum'),
                                                  sum_diff_y_micron=('diff_y_micron', 'sum'),

                                                  f_area_micron=('area_micron', 'mean'),
                                                  f_perimeter_au_norm=('perimeter_au_norm', 'mean'),
                                                  # morphology

                                                  f_min_dist_micron=('min_dist_micron', 'mean'),
                                                  # minimal distance to edge averaged for each timelapse
                                                  min_min_dist_micron=('min_dist_micron', 'min'),
                                                  max_min_dist_micron=('min_dist_micron', 'max'),
                                                  beg_min_dist_micron=('min_dist_micron', 'first'),
                                                  end_min_dist_micron=('min_dist_micron', 'last'),
                                                  f_var_dist_micron=('min_dist_micron', 'var'),
                                                  )

data_agg['f_Rvar_diff_xy_micron'] = data_agg['f_var_diff_xy_micron'] / data_agg['f_mean_diff_xy_micron']
data_agg['f_Rvar_dist_micron'] = data_agg['f_var_dist_micron'] / data_agg['f_min_dist_micron']
# Relative variance

data_agg['f_total_displacement'] = np.sqrt((data_agg['sum_diff_x_micron']) ** 2 + (data_agg['sum_diff_y_micron']) ** 2)
# distance from first to last coordinate
data_agg['f_persistence'] = data_agg['f_total_displacement'] / data_agg['f_sum_diff_xy_micron']
# shows how directional the movement is

data_agg['file_mean_diff_xy_micron'] = data_agg.groupby('file')['f_mean_diff_xy_micron'].transform(np.max)
data_agg['f_fastest_mask'] = np.where((data_agg['f_mean_diff_xy_micron'] == data_agg['file_mean_diff_xy_micron']), 1, 0)
# DO NOT USE FOR guide AS TARGET (telo!)
# the fastest (or the only available) dot in the nucleus is 1, the rest is 0

data_agg['f_min_dist_range'] = data_agg['max_min_dist_micron'] - data_agg['min_min_dist_micron']
# min_dist change within timelapse (max-min) for each dot
data_agg['f_total_min_dist'] = data_agg['end_min_dist_micron'] - data_agg['beg_min_dist_micron']
# how distance changed within timelapse (frame29-frame0)

data_agg['file_max_min_dist_micron'] = data_agg.groupby('file')['f_min_dist_micron'].transform(np.max)
data_agg['f_most_central_mask'] = np.where((data_agg['f_min_dist_micron'] == data_agg['file_max_min_dist_micron']), 1,
                                           0)
# DO NOT USE FOR guide AS TARGET (telo!)
# the most central (or the only available) dot in the nucleus is 1, the rest is 0

data_slope = data.groupby(['file', 'particle']).apply(lambda x: linregress(x['frame'], x['min_dist_micron'])[0])
data_agg['f_slope_min_dist_micron'] = data_slope
# slope for minimal distance to edge; how distance to edge changes within the timelapse?


data_slope_area = data.groupby(['file', 'particle']).apply(lambda x: linregress(x['frame'], x['area_micron'])[0])
data_agg['f_slope_area_micron'] = data_slope_area
# slope for nucleus area; how area changes within the timelapse?

data_slope_perimeter = data.groupby(['file', 'particle']).apply(lambda x: linregress(x['frame'],
                                                                                     x['perimeter_au_norm'])[0])
data_agg['f_slope_perimeter_au_norm'] = data_slope_perimeter
# slope for nucleus perimeter

data_SD_diff_xy_micron = data.groupby(['file', 'particle']).agg(SD_diff=('diff_xy_micron', 'std'))
data_i = data.set_index(['file', 'particle'])
data_i['SD_diff_xy_micron'] = data_SD_diff_xy_micron
data_i['f_mean_diff_xy_micron'] = data_agg['f_mean_diff_xy_micron']
data_i['outliers2SD_diff_xy'] = np.where((data_i['diff_xy_micron'] >
                                          (data_i['f_mean_diff_xy_micron'] + 2 * data_i['SD_diff_xy_micron'])), 1, 0)
data_i['outliers3SD_diff_xy'] = np.where((data_i['diff_xy_micron'] >
                                          (data_i['f_mean_diff_xy_micron'] + 3 * data_i['SD_diff_xy_micron'])), 1, 0)
data_agg['f_outliers2SD_diff_xy'] = data_i.groupby(['file', 'particle']) \
    .agg(f_outliers2SD_diff_xy=('outliers2SD_diff_xy', 'sum'))
data_agg['f_outliers3SD_diff_xy'] = data_i.groupby(['file', 'particle']) \
    .agg(f_outliers3SD_diff_xy=('outliers3SD_diff_xy', 'sum'))
# is there a displacement larger than mean plus 2SD or 3SD (SD calculated for each dot, 29xy pairs) respectively

data_sterile = data_agg.drop(['sum_diff_x_micron',
                              'sum_diff_y_micron',
                              'min_min_dist_micron',
                              'max_min_dist_micron',
                              'beg_min_dist_micron',
                              'end_min_dist_micron',
                              'file_mean_diff_xy_micron',
                              'file_max_min_dist_micron',
                              'f_sum_diff_xy_micron',  # proportional to f_mean_diff_xy_micron, thus, useless
                              ], axis=1)
data_sterile.reset_index(inplace=True)
corr_features = data_sterile.corr()
# cleaning up

# data_sterile = pd.read_csv('scripts/data_sterile_PCA_92ba95d.csv')
# features = data_sterile.columns[7:]
# to get data with principal components

''' 
Train / test split
!!! train/test split is deprecated, cross-val with whole dataset is used from now on

tst = int((data_sterile['file'].unique().shape[0]) / 5)
# nuclei number to choose for testing

test_choice = np.random.RandomState(4242).choice(data_sterile['file'].unique(), tst, replace=False)
test_data = data_sterile[data_sterile['file'].isin(test_choice)]
train_data = data_sterile[~data_sterile['file'].isin(test_choice)]
train_data = train_data.dropna()

X = train_data[features]
y = train_data['t_serum_conc_percent'].astype('str')
'''

features = [
    'f_mean_diff_xy_micron', 'f_max_diff_xy_micron', 'f_var_diff_xy_micron',
    'f_area_micron', 'f_perimeter_au_norm', 'f_min_dist_micron',
    'f_var_dist_micron', 'f_Rvar_diff_xy_micron', 'f_Rvar_dist_micron',
    'f_total_displacement', 'f_persistence', 'f_fastest_mask',
    'f_min_dist_range', 'f_total_min_dist', 'f_most_central_mask',
    'f_slope_min_dist_micron', 'f_slope_area_micron',
    'f_slope_perimeter_au_norm', 'f_outliers2SD_diff_xy',
    'f_outliers3SD_diff_xy'
]

X = data_sterile[features]
y = data_sterile['t_serum_conc_percent']  # .astype('str')
y = (y / 10).astype('int')  # '10% serum' = 1, '0.3% serum' = 0

''' 
Baseline performance estimation


tree = DecisionTreeClassifier(random_state=0, max_depth=1)
gkf = GroupKFold(n_splits=4)

metrics = ['accuracy', 'precision', 'recall', 'f1']
baseline_scores = pd.DataFrame()
for metric in metrics:
    trees = pd.DataFrame()
    for feature in features:
        trees[feature] = cross_val_score(tree, X[feature].values.reshape(-1, 1), y,  # worked, now returns index error
                                         cv=gkf, groups=data_sterile['file'],
                                         scoring=metric
                                         )

    trees = trees.transpose()
    baseline_scores[metric] = trees.mean(axis=1)
'''
# since 1-level decision tree cannot overfit, we can train and score without cross-validation?

'''
trees_noCV = []
for feature in features:
    tree_noCV = DecisionTreeClassifier(random_state=0, max_depth=1)
    trees_noCV[feature] = tree_noCV.fit(X[feature].values.reshape(-1, 1), y).score(X[feature].values.reshape(-1, 1), y)

trees_noCV = trees_noCV.transpose()

'''

''' 
Random forest
'''

'''
forest = RandomForestClassifier(n_estimators=1000, max_features=2, random_state=42)
gkf = GroupKFold(n_splits=3)
print("Cross-validation scores:\n{}".format(cross_val_score(forest, X, y, cv=gkf, groups=train_data['file'])))

param_grid = {'max_features': [1, 2, 3, 5, 10, 15, "auto"],
              'max_depth': [1, 2, 3, 5, 10, None]}

grid_forest = RandomForestClassifier(n_estimators=1000)
grid_search = GridSearchCV(grid_forest, param_grid, cv=gkf)
grid_search.fit(X, y, groups=train_data['file'])
grid_forest_results = pd.DataFrame(grid_search.cv_results_)
forest_importances = grid_search.best_estimator_.feature_importances_
std = np.std([tree.feature_importances_ for tree in grid_search.best_estimator_.estimators_], axis=0)
var = np.var([tree.feature_importances_ for tree in grid_search.best_estimator_.estimators_], axis=0)
'''

''' 
Gradient boosting trees
'''

# boosted_forest = GradientBoostingClassifier(n_estimators=1000, random_state=0)
gkf = GroupKFold(n_splits=4)
# print("Cross-validation scores:\n{}".format(cross_val_score(boosted_forest, X, y, cv=gkf, groups=train_data['file'])))

b_param_grid = {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                'max_depth': [4, 5, 6, 7, 8, 9, 10, 20]}

grid_b_forest = GradientBoostingClassifier(n_estimators=1000)
b_grid_search = GridSearchCV(grid_b_forest, b_param_grid, cv=gkf,
                             scoring=['accuracy', 'precision', 'recall', 'f1'],
                             refit=False)
b_grid_search.fit(X, y, groups=data_sterile['file'])
b_grid_forest_results = pd.DataFrame(b_grid_search.cv_results_)

# forest_b_importances = b_grid_search.best_estimator_.feature_importances_

# forests_comp = pd.DataFrame({'RF': forest_importances, 'GBC': forest_b_importances}, index=X.columns)
# forests_comp.plot(kind='bar')
# compare importances for forest types

# y.value_counts() # to check if classes are balanced
# b_grid_forest_results.to_csv('C:/Users/redchuk/python/temp/temp_n_track_RF/boosted_forest_results.csv')

b_pvt = pd.pivot_table(b_grid_forest_results,
                       values='mean_test_accuracy',
                       index='param_learning_rate',
                       columns='param_max_depth')
sns.heatmap(b_pvt, annot=True)
plt.show()
plt.close()

'''
rf_pvt = pd.pivot_table(grid_forest_results,
                        values='mean_test_score',
                        index='param_max_features',
                        columns='param_max_depth')
'''

''' 
Plotting forests results
'''

# data_to_predict = data_sterile[features]
# data_sterile['predicted_boosted'] = b_grid_search.best_estimator_.predict(data_sterile[features]).astype(float)
# data_sterile.to_csv('C:/Users/redchuk/python/temp/temp_n_track_RF/boosted_predict66.csv')

''' 
Custom-made leave-one-group-out
'''
'''
data_sterile = pd.read_csv('scripts/data_sterile_PCA_92ba95d.csv')
data_sterile.set_index(['file', 'particle'], drop=False, inplace=True)
features = data_sterile.columns[7:]
boosted_forest = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.001, max_depth=4, random_state=62)
gkf = GroupKFold(n_splits=153)

# loo = cross_val_score(boosted_forest, X, y, cv=gkf, groups=data_sterile['file'])
# np.mean(loo)
# this is not weighted by number of samples, so I'll need manual LOO to predict

for inx in data_sterile['file'].unique():
    train_data = data_sterile[~data_sterile['file'].isin([inx])]
    test_data = data_sterile[data_sterile['file'].isin([inx])]
    X = train_data[features]
    y = train_data['t_serum_conc_percent']  # .astype('str')
    y = (y / 10).astype('int')  # '10% serum' = 1, '0.3% serum' = 0
    boosted_forest.fit(X, y)
    for ii in test_data.index:
        predicted = boosted_forest.predict(test_data.loc[ii, features].values.reshape(1, -1))[0]
        data_sterile.loc[ii, '5c1e1b74_gbc_predicted'] = predicted
        print(predicted)

comp = ((data_sterile['t_serum_conc_percent']/ 10).astype('int')==data_sterile['5c1e1b74_gbc_predicted'])
sum(comp)/302
#  this gives accuracy 0.5529801324503312 for LOOCV,
#  which is lower than accuracy estimated by (repeated) K-foldCV (k=4),
#  discussed by others here
#  https://stats.stackexchange.com/questions/61783/bias-and-variance-in-leave-one-out-vs-k-fold-cross-validation
'''

'''
classification_report
'''

param_from_gs = [(10, 10), (0.0001, 10), (1, 6)]  # (learning_rate, max_depth)
predictions = []
reports = []
for pair in param_from_gs:
    gbc = GradientBoostingClassifier(n_estimators=1000, learning_rate=pair[0], max_depth=pair[1])
    gkf = GroupKFold(n_splits=4)
    cv_pred = cross_val_predict(gbc, X, y, cv=gkf, groups=data_sterile['file'])
    predictions.append(cv_pred)
    report = classification_report(y, cv_pred)
    reports.append(report)
    print(f'learning_rate={pair[0]}, max_depth={pair[1]}')
    print(report)

'''
SHAP
'''

X = data_sterile[features]
y = data_sterile['t_serum_conc_percent']  # .astype('str')
y = (y / 10).astype('int')  # '10% serum' = 1, '0.3% serum' = 0

#gss = GroupShuffleSplit(n_splits=4)

for strain, stest in gkf.split(X, y, data_sterile['file']):
#for strain, stest in gss.split(X, y, data_sterile['file']):
    train_data = data_sterile.iloc[strain,:]
    test_data = data_sterile.iloc[stest,:]
    sX = train_data[features]
    sy = train_data['t_serum_conc_percent']
    sy = (sy / 10).astype('int')
    sX_test = test_data[features]
    sy_test = test_data['t_serum_conc_percent']
    sy_test = (sy_test / 10).astype('int')

    gbc = GradientBoostingClassifier(n_estimators=1000, learning_rate=10, max_depth=10)
    gbc.fit(sX, sy)

    explainer = shap.TreeExplainer(gbc)
    shap_values = explainer.shap_values(sX_test)
    shap.summary_plot(shap_values, sX_test, plot_size=(25,7))




