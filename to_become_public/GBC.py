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
grid_iterations = 2  # todo: remove before flight
cv_iterations = 2  # todo: remove before flight
sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True)
groups = indexed['file']

''' GBC grid search '''

pivots = []
baselines = []

for i in range(grid_iterations):
    param_grid = {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]}

    gbc = GradientBoostingClassifier(n_estimators=1000)
    grid_search = GridSearchCV(gbc, param_grid, cv=sgkf, refit=False)

    grid_search.fit(X, y, groups=groups)

    pivots.append(pd.pivot_table(pd.DataFrame(grid_search.cv_results_),
                           values='mean_test_score',
                           index='param_learning_rate',
                           columns='param_max_depth'))

    baseline = DecisionTreeClassifier(max_depth=1)
    baselines.append(np.mean(cross_val_score(baseline, X, y, cv=sgkf, groups=groups)))

mean_accuracy = pd.concat(pivots).mean(level=0)
sns.heatmap(mean_accuracy, annot=True) # todo: how to treat plotting?
plt.show()
plt.close()

print('Baseline performance: ' + str(np.mean(baselines)))  # todo: how to treat printing?

best_max_depth = mpvts.max().idxmax()
best_learning_rate = mpvts.max(axis=1).idxmax()

''' GBC SHAP '''

ix = 0
shap_repeats = pd.DataFrame()

for i in range(cv_iterations):
    print('shap iter' + str(i))  # todo: remove before flight
    shap_vs_list = []
    sX_test_list = []
    s_id_list = []

    idx_file_particle = indexed[['file', 'particle']]

    for train_idxs, test_idxs in sgkf.split(X, y, groups):
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
