import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from matplotlib import pyplot as plt
from to_become_public.feature_engineering import get_data  # todo: correct before publishing
import shap

X, y, indexed = get_data('to_become_public/tracking_output/data_47091baa.csv')
cv_iterations = 2
gkf = StratifiedGroupKFold(n_splits=4, shuffle=True)

''' GBC '''

pivots = []
grids = []
baselines = []

for i in range(cv_iterations):
    b_param_grid = {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]}

    grid_b_forest = GradientBoostingClassifier(n_estimators=1000)
    b_grid_search = GridSearchCV(grid_b_forest, b_param_grid, cv=gkf, refit=False)

    b_grid_search.fit(X, y, groups=indexed['file'])
    grids.append(b_grid_search.cv_results_)

    b_pvt = pd.pivot_table(pd.DataFrame(b_grid_search.cv_results_),
                           values='mean_test_score',
                           index='param_learning_rate',
                           columns='param_max_depth')

    pivots.append(b_pvt)

    tree = DecisionTreeClassifier(max_depth=1)
    baselines.append(np.mean(cross_val_score(tree, X, y, cv=gkf, groups=indexed['file'])))

mpvts = pd.concat(pivots).mean(level=0)
sns.heatmap(mpvts, annot=True)
plt.title(str(cv_iterations) + ' cv reps')
plt.show()
plt.close()

print('baseline performance: ' + str(np.mean(baselines)))  # todo: how to treat printing?

best_max_depth = mpvts.max().idxmax()
best_learning_rate = mpvts.max(axis=1).idxmax()

''' GBC SHAP '''

ix = 0
shap_repeats = pd.DataFrame()

for i in range(cv_iterations):

    shap_vs_list = []
    sX_test_list = []
    s_id_list = []

    groups = indexed['file']
    idx_file_particle = indexed[['file', 'particle']]

    for train_idxs, test_idxs in gkf.split(X, y, groups):
        X_train = X.loc[train_idxs]
        X_test = X.loc[test_idxs]
        y_train = y.loc[train_idxs]
        y_test = y.loc[test_idxs]

        gbc = GradientBoostingClassifier(n_estimators=1000, learning_rate=best_learning_rate, max_depth=best_max_depth)
        gbc.fit(X_train, y_train)

        explainer = shap.TreeExplainer(gbc)
        shap_values = explainer.shap_values(X_test)

        shap_vs_list.append(shap_values)
        sX_test_list.append(X_test)
        s_id_list.append(idx_file_particle.loc[test_idxs])

        '''
        shap.summary_plot(shap_values,
                          X_test,
                          feature_names=X.columns,
                          sort=False,
                          color_bar=False,
                          plot_size=(10, 10),
                          )
        '''

    all_sX_test = pd.concat(sX_test_list)
    all_splits_shap = np.concatenate(shap_vs_list)
    df_all_splits_shap = pd.DataFrame(all_splits_shap, columns=all_sX_test.columns).add_prefix('shap_')
    all_s_id = pd.concat(s_id_list)

    '''
    plt.title('Aggregated from 4CV splits')
    shap.summary_plot(all_splits_shap,
                      all_sX_test,
                      feature_names=X.columns,
                      sort=False,
                      color_bar=False,
                      plot_size=(10, 10),
                      )
    '''

    list_to_concat = [all_sX_test.reset_index(),
                      df_all_splits_shap.reset_index(),
                      all_s_id.reset_index()]

    df_all = pd.concat(list_to_concat, axis=1) \
        .drop('index', axis=1).set_index(['file', 'particle']).add_prefix(str(i) + 'r_')

    shap_repeats = shap_repeats.join(df_all) if not shap_repeats.empty else df_all

shap_averaged = pd.DataFrame()

for i in df_all.columns:
    shap_averaged[i] = shap_repeats[shap_repeats.columns[range(df_all.columns.tolist().index(i),
                                                               len(shap_repeats.columns),
                                                               40)]].mean(axis=1)

shap.summary_plot(shap_averaged.iloc[:, 20:].to_numpy(),
                  shap_averaged.iloc[:, :20].to_numpy(),
                  feature_names=X.columns,
                  sort=False,
                  color_bar=False,
                  plot_size=(10, 10),
                  )
