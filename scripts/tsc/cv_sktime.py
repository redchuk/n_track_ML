import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import StratifiedGroupKFold

import warnings
warnings.simplefilter(action='ignore', category=sklearn.exceptions.UndefinedMetricWarning)

from etl_tsc import load_data, get_X_dfX_y_groups, fsets

#logger = None

scoring=['accuracy', 'precision','recall','f1']


def format_scores_df(scores):
    scores = scores.drop(columns=['fit_time','score_time'])
    scores.columns = scores.columns.str.replace('test_','')
    #print(scores.mean())
    #print(scores.std())
    return scores

def cv_sgkf(classifier, X, y, groups, repeats=10):
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True)
    #print(classifier)
    #print(cv)

    scores_all = []
    for i in range(repeats):
        #cv = StratifiedGroupKFold(n_splits=4, shuffle=True)
        scores = cross_validate(classifier, X, y, cv=cv, scoring=scoring, groups=groups)
        scores_all.append(pd.DataFrame.from_dict(scores))
    
    scores = pd.concat(scores_all)
    scores = format_scores_df(scores)
    score = scores['cv'] = 'StratifiedGroupKFold'
    return scores


# In[ ]:


from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

def run_kneighbors(data, n_neighbors=5, repeats=20):
    logger.info("run_kneighbors")
    classifier = KNeighborsTimeSeriesClassifier(n_neighbors=n_neighbors)

    scores_all = []
    for fset in fsets.keys():
        logger.debug("run_kneighbors: fset: " + fset)
        #X, dfX, y, groups, debug_polar, debug_data = prepare_Xy(data, fset)
        X, dfX, y, groups, debugm, debugn = get_X_dfX_y_groups(data, fset)

        logger.debug("run_kneighbors: shape X: " + str(X.shape))
        logger.debug("run_kneighbors: shape y: " + str(y.shape))
        logger.debug("run_kneighbors: shape groups: " + str(groups.shape))
        
        scores = cv_sgkf(classifier, X, y, groups, repeats=repeats)
        scores['classifier'] = 'KNeighborsTSC'
        scores['fset'] = fset
        scores_all.append(scores)

    scores = pd.concat(scores_all)
    print("run_kneighbors")
    print(scores.groupby(['cv','fset']).mean())
    print(scores.groupby(['cv','fset']).std())
    print(scores.shape)

    return scores



# In[ ]:


from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sktime.transformations.panel.rocket import Rocket

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def run_rocket(data, repeats=20):
    logger.info("run_rocket")
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
    #classifier = RidgeClassifier()
    #classifier.fit(X_transform, y)


    scores_all = []
    for fset in fsets.keys():
        logger.debug("run_rocket: fset: " + fset)
        #X, dfX, y, groups, debugm, debugn = prepare_Xy(data, fset)
        X, dfX, y, groups, debugm, debugn = get_X_dfX_y_groups(data, fset)

        logger.debug("run_rocket: shape dfX: " + str(dfX.shape))
        logger.debug("run_rocket: shape y: " + str(y.shape))
        logger.debug("run_rocket: shape groups: " + str(groups.shape))
        rocket = Rocket()  # by default, ROCKET uses 10,000 kernels
        rocket.fit(dfX)
        X_transform = rocket.transform(dfX)

        scores = cv_sgkf(classifier, X_transform, y, groups, repeats=repeats)
        scores['classifier'] = 'Rocket'
        scores['fset'] = fset
        scores_all.append(scores)

    scores = pd.concat(scores_all)
    
    print("run_rocket")
    print(scores.groupby(['cv','fset']).mean())
    print(scores.groupby(['cv','fset']).std())
    print(scores.shape)

    return scores



# 
# 
# 1.   Stratified split (add function, with loop)
# 2.   Infinity loop to check epoch effect for inceptiontime
# 3.   SHAP for NN, TSC
# 4.   plot acc (mean, std)
# 
# 

# In[ ]:



import click
import pandas as pd
from pathlib import Path

from utility import parse_config, set_logger

@click.command()
#@click.argument("config_file", type=str, default="config.yml")
@click.option("--config_paths", type=str, default="paths.yml")
@click.option("--config_tsc", type=str, default="config.yml")
@click.option("--ext", type=str, default=".csv")
def cv_sktime(config_paths, config_tsc, ext):
    paths = parse_config(config_paths)
    config = parse_config(config_tsc)

    log_dir = paths["log"]["tsc"]
    
    # configure logger
    global logger
    logger = set_logger(log_dir + "/cv_sktime.log")

    # Load config from config file
    logger.info(f"Load config from {config_tsc}")


    # read the data 
    data_dir = paths['data']
    raw_data_file = config["etl"]["raw_data_file"]

    data = load_data(Path(data_dir) / raw_data_file)
    logger.info('Loaded data shape: ' + str(data.shape))


    scores_all = []
    
    # run KNeighbors
    n_neighbors = config["kneighbors"]["n_neighbors"]
    repeats = config["kneighbors"]["repeats"]
    scores = run_kneighbors(data, n_neighbors, repeats)
    scores_all.append(scores)

    # run Rocket
    repeats = config["rocket"]["repeats"]
    scores = run_rocket(data, repeats)
    scores_all.append(scores)


    # store output
    output_dir = paths["output"]["cv"]
    output_csv = config["output"]["csv"]
    scores = pd.concat(scores_all)
    scores.to_csv(Path(output_dir) / output_csv, index=False)
    
    
if __name__ == "__main__":
    cv_sktime()

    
