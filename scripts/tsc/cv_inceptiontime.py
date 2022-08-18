import keras
from pathlib import Path
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from utility import parse_config
from etl_tsc import load_data, get_X_dfX_y_groups, fsets

import logging
logger = logging.getLogger(__name__)

'''
This method was copied and modified from 'prepare_data' in InceptionTime/main.py:
https://github.com/hajaalin/InceptionTime/blob/f3fd6c5e9298ec9ca5d0fc594bb07dd1decc3718/main.py#L15
'''
def prepare_data_for_inception(X,y):
    # use all data, use cross-validation
    x_train = X.copy()
    y_train = y.copy()

    nb_classes = len(np.unique(np.concatenate((y_train,), axis=0)))

    # make the min to zero of labels
    #y_train, y_test = transform_labels(y_train, y_test)

    # save orignal y because later we will use binary
    y_true_train = y_train.astype(np.int64)
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train,), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, nb_classes, y_true_train, enc


from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score


'''
Single cross-validation run
'''
def inceptiontime_cv(cv, X_inc, y_inc, y_true, groups, output_it, \
                     kernel_size, epochs=250, nb_classes=2):
    output_directory = output_it
    input_shape = X_inc.shape[1:]
    verbose = False

    from classifiers import inception
    classifier_keras = None
    classifier_keras = inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, \
                                                      kernel_size=kernel_size, nb_epochs=epochs, \
                                                      verbose=verbose)
    def create_model():
        #print(classifier_keras.model)
        #classifier_keras.model.summary()
        return classifier_keras.model

    batch_size = int(min(X_inc.shape[0] / 10, 16))
    columns = ['accuracy','precision','recall','f1']
    scores = pd.DataFrame(columns=columns)

    # One-hot encoding is a problem for StratifiedGroupKFold,
    # split using y_true
    for train_index,val_index in cv.split(X_inc,y_true,groups):
        #print('cv loop')
        #print(train_index.shape)
        #print(X_inc[train_index].shape)
        input_shape = X_inc[train_index].shape[1:]
        #print(input_shape)

        #print(train_index)
        #print(val_index)
        #continue
        #print(y_true[train_index])
        #print(y_inc[train_index])
        #break

        #classifier_keras = inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes,                                                           nb_epochs=epochs, verbose=verbose)
        classifier = KerasClassifier(model=create_model, \
                                     epochs=epochs, \
                                     batch_size=batch_size, \
                                     verbose=verbose)
        classifier.fit(X_inc[train_index],y_inc[train_index])
        pred = classifier.predict(X_inc[val_index])

        truth = y_true[val_index]
        #print('truth')
        #print(truth)
        #print('pred')
        #print(pred)

        # prediction is onehot-encoded, reverse it
        pred = pred.argmax(1)
        #print(pred)

        # get fold accuracy and append
        fold_acc = accuracy_score(truth, pred)
        fold_prc = precision_score(truth, pred)
        fold_rec = recall_score(truth, pred)
        fold_f1 = f1_score(truth, pred)
        scores.loc[len(scores)] = [fold_acc,fold_prc,fold_rec,fold_f1]
    
    scores['classifier'] = 'InceptionTime'
    scores['kernel_size'] = kernel_size
    scores['epochs'] = epochs

    return scores


'''
Repeat cross-validation
'''
def inceptiontime_cv_repeat(data, output_it, fset, kernel_size=20, epochs=250, repeats=10):
    logger.info(fset)
    X, dfX, y, groups, debugm, debugn = get_X_dfX_y_groups(data, fset)

    # prepare_data_for inception returns all, no split to train and test sets
    X_inc, y_inc, nb_classes, y_true, enc = prepare_data_for_inception(X,y)

    logger.debug("X_inc: " + str(X_inc.shape))
    logger.debug("y_inc: " + str(y_inc.shape))

    cv = StratifiedGroupKFold(n_splits=4, shuffle=True)

    scores_all = []
    for i in range(repeats):
        logger.debug('repeat: %d/%d' % (i+1, repeats))
        scores = inceptiontime_cv(cv, X_inc, y_inc, y_true, groups, output_it, \
                                  kernel_size=kernel_size, epochs=epochs, \
                                  nb_classes=nb_classes)
        scores['repeat'] = i+1
        scores_all.append(scores)
    scores = pd.concat(scores_all)

    scores['cv'] = str(cv)
    scores['fset'] = fset
    scores['kernel_size'] = kernel_size
    scores['epochs'] = epochs   

    logger.info(f"inceptiontime_cv_repeat: %s feature_set:%s, kernel_size:%d, epochs:%d, accuracy:%f0.00" % (str(cv), fset, kernel_size, epochs, scores['accuracy'].mean()))
    
    return scores


import click
from datetime import datetime
import pandas as pd
import sys
import time

@click.command()
@click.option("--paths", type=str, default="paths.yml")
@click.option("--kernel_size", type=int, default=20)
@click.option("--epochs", type=int, default=100)
@click.option("--fset", type=click.Choice(["f_mot","f_mot_morph","f_mot_morph_dyn"]), default="f_mot_morph")
@click.option("--loop_fsets", is_flag=True, default=False)
@click.option("--repeats", type=int, default=20)
@click.option("--job_name", type=str, default="tsc_it")
@click.option("--job_id", type=str)
@click.option("--now", type=str)
def cv_inceptiontime(paths, kernel_size, epochs, fset, loop_fsets, repeats, job_name, job_id, now):
    paths = parse_config(paths)

    log_dir = Path(paths["log"]["tsc"]) / job_name / now
    log_dir.mkdir(parents=True, exist_ok=True)

    if not now:
        now = datetime.now().strftime("%Y%m%d%H%M%S")

    # configure logger
    log_file = log_dir / (f"cv_inceptiontime_%s_%s_%s.log" % (job_name, now, job_id))
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Finished logger configuration!")
    print("Logging to " + str(log_file))


    # add InceptionTime source to Python path
    src_inceptiontime = paths["src"]["inceptiontime"]
    sys.path.insert(1, src_inceptiontime)

    # output folders
    output_cv = Path(paths["output"]["cv"]) / job_name / now
    output_cv.mkdir(parents=True, exist_ok=True)
    output_it = Path(paths["output"]["it"]) / job_id
    output_it.mkdir(parents=True, exist_ok=True)
    output_it = str(output_it) + "/"

    # read the data 
    data_dir = paths["data"]["dir"]
    raw_data_file = paths["data"]["raw_data_file"]

    data = load_data(Path(data_dir) / raw_data_file)
    logger.info('Loaded data shape: ' + str(data.shape))


    tic = time.perf_counter()
    
    scores_all = []

    if loop_fsets:
        logger.info("loop feature sets")

        for f in fsets.keys():
            scores = inceptiontime_cv_repeat(data, output_it, f, kernel_size=kernel_size, epochs=epochs, repeats=repeats)
            scores_all.append(scores)

    else:
        logger.info("single fset")
        scores = inceptiontime_cv_repeat(data, output_it, fset, kernel_size=kernel_size, epochs=epochs, repeats=repeats)
        scores_all.append(scores)
        
    toc = time.perf_counter()
    logger.info(f"Finished processing in {(toc-tic) / 60:0.1f} minutes.")
            
    scores = pd.concat(scores_all)

    scores_file = "cv_" + job_name + "_k" + str(kernel_size) + "_e" + str(epochs) + "_" + now + ".csv"
    scores_file = output_cv / scores_file
    scores.to_csv(scores_file, index=False)
    logger.info("Wrote scores to " + str(scores_file))

if __name__ == "__main__":
    cv_inceptiontime()
    

