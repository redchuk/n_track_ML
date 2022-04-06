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
@click.option("--config_paths", type=str, default="scripts/tsc/paths.yml")
@click.option("--config_tsc", type=str, default="scripts/tsc/config.yml")
@click.option("--loop_epochs", is_flag=True, default=False)
@click.option("--loop_fsets", is_flag=True, default=False)
def cv_inceptiontime(config_paths, config_tsc, loop_epochs, loop_fsets):
    paths = parse_config(config_paths)
    config = parse_config(config_tsc)

    log_dir = paths["log"]["tsc"]
    
    now = datetime.now().strftime("%Y-%m%d-%H%M")

    # configure logger
    log_file = log_dir + "/cv_inceptiontime_" + now + ".log"
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Finished logger configuration!")
    print("Logging to " + log_file)


    # add InceptionTime source to Python path
    src_inceptiontime = paths["src"]["inceptiontime"]
    sys.path.insert(1, src_inceptiontime)

    # output folders
    output_cv = paths["output"]["cv"]
    output_it = paths["output"]["it"]

    # read the data 
    data_dir = paths['data']
    raw_data_file = config["etl"]["raw_data_file"]

    data = load_data(Path(data_dir) / raw_data_file)
    logger.info('Loaded data shape: ' + str(data.shape))

    kernel_size = config["inceptiontime"]["kernel_size"]
    repeats = config["inceptiontime"]["repeats"]


    tic = time.perf_counter()
    
    scores_all = []

    if loop_epochs:
        logger.info("loop epochs")
        scores_file = "loop_epochs_k" + str(kernel_size) + "_" + now + ".csv"
        fset = "f_mot_morph"
        for k in range(5, 600, 10):
            scores = inceptiontime_cv_repeat(data, output_it, fset, epochs=k, repeats=repeats)
            scores_all.append(scores)

    elif loop_fsets:
        logger.info("loop feature sets")
        epochs = config["inceptiontime"]["epochs"]
        scores_file = "inceptiontime_k" + str(kernel_size) + "_e" + str(epochs) + "_" + now + ".csv"
        for fset in fsets.keys():
            scores = inceptiontime_cv_repeat(data, output_it, fset, epochs=epochs, repeats=repeats)
            scores_all.append(scores)

    else:
        logger.info("single fset")
        epochs = config["inceptiontime"]["epochs"]
        fset = config["inceptiontime"]["fset"]
        scores_file = "inceptiontime_k" + str(kernel_size) + "_e" + str(epochs) + "_" + now + ".csv"
        scores = inceptiontime_cv_repeat(data, output_it, fset, epochs=epochs, repeats=repeats)
        scores_all.append(scores)
        
    toc = time.perf_counter()
    logger.info(f"Finished processing in {(toc-tic) / 60:0.1f} minutes.")
    
    # store output
    output_dir = paths["output"]["cv"]
    #output_csv = config["output"]["csv"]
    scores = pd.concat(scores_all)


    scores_file = Path(output_dir) / scores_file
    scores.to_csv(scores_file, index=False)
    logger.info("Wrote scores to " + str(scores_file))

if __name__ == "__main__":
    cv_inceptiontime()
    

