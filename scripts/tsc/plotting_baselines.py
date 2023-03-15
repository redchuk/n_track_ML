import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from matplotlib import pyplot as plt
#from feature_engineering import get_data  # correct version
from to_become_public.feature_engineering import get_data  # todo: correct before publishing (Pycharm only)
import shap
import pandas as pd

#path = 'tracking_output/data_47091baa.csv'# correct version
path = 'to_become_public/tracking_output/data_47091baa.csv'  # todo: correct before publishing (Pycharm only)
#outpath = 'shap_averaged_GBC.csv' # correct version
outpath = 'to_become_public/tracking_output/shap_averaged_GBC.csv'  # todo: correct before publishing (Pycharm only)
data_from_csv = pd.read_csv(path)
X, y, indexed = get_data(data_from_csv)

base_accs = pd.DataFrame()
for i in range(20):
    print(i)
    accdict = {}
    for feature in X.columns:
        gkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=None)
        # acc = cross_val_score(tree, X[feature].values.reshape(-1, 1), y, cv=gkf, groups=data_sterile['file'])
        # print('baseline ('+feature+'): ' + str(np.mean(acc)))
        score_test = []
        score_train = []

        for strain, stest in gkf.split(X, y, indexed['file']):
            train_data = indexed.iloc[strain, :]
            test_data = indexed.iloc[stest, :]
            sX = train_data[feature].values.reshape(-1, 1)
            sy = train_data['serum']

            sX_test = test_data[feature].values.reshape(-1, 1)
            sy_test = test_data['serum']

            tree = DecisionTreeClassifier(max_depth=1)
            tree.fit(sX, sy)
            score_train.append(tree.score(sX, sy))
            # print(tree.score(sX, sy))
            score_test.append(tree.score(sX_test, sy_test))
            # print(tree.score(sX_test, sy_test))
        accdict[feature] = np.mean(score_test)
    #print(feature + ': train ' + str(np.mean(score_train)) + ' test ' + str(np.mean(score_test)))
    base_accs[i] = pd.Series(accdict)

base_accs['mean'] = base_accs.mean(axis=1)
base_accs['base_sf_rank']=np.argsort(np.argsort(base_accs['mean']))
base_accs.to_csv('data/20230315_02f404cc_acc_1lvlTREE.csv', index=False)

