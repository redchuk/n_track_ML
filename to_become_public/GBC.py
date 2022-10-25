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

X, y, indexed = get_data('to_become_public/tracking_output/data_47091baa.csv')

cv_iterations = 1
gkf = StratifiedGroupKFold(n_splits=4, shuffle=True)

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
pvt_sem = pd.concat(pivots).sem(level=0)  # standard error of mean, element-wise
sns.heatmap(mpvts, annot=True)
plt.title(str(cv_iterations) + ' cv reps')
plt.show()
plt.close()
