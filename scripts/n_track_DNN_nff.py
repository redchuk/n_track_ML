'''
unlike in n_track_DNN, here the files with <30 frames are dropped
somehow this gives +2% acc
'''




import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from keras import models
from keras import layers
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from datetime import datetime
import matplotlib.pyplot as plt
import shap

''' 
read the data 
'''
data = pd.read_csv('data/a286935_data_chromatin_live.csv')
data = data[~data["comment"].isin(["stress_control"])]
data = data[~data["comment"].isin(["H2B"])]
data = data[data["guide"].str.contains('1398') | data["guide"].str.contains('1514')]
data = data[data["time"] < 40]
data.set_index(['file', 'particle'], inplace=True)
# original data before aggregation
fr_in_ind = data.index.value_counts()
dots_to_drop = fr_in_ind[fr_in_ind < 30]
data.drop(dots_to_drop.index, inplace=True)
# check for dots with less than 30 frames
data.set_index('frame', append=True, inplace=True)
data = data[['diff_xy_micron', 'area_micron', 'perimeter_au_norm', 'min_dist_micron']]
# filter features
data = data.unstack()  # reshape
data.drop(('diff_xy_micron', 0), axis=1, inplace=True)  # drop first delta 'diff_xy_micron', which is NaN
data.columns = data.columns.to_flat_index()  # needed for concat
data_sterile = pd.read_csv('data/data_sterile_PCA_92ba95d.csv')
data_sterile.set_index(['file', 'particle'], inplace=True)
features = data_sterile.columns[4:]
data_sterile = data_sterile[features]
# data after aggregation and feature engineering
data_sterile.drop(dots_to_drop.index, inplace=True)
# drop those with less than 30 frames
data_raw = data_sterile.join(data)
# concatenate

data_sterile = data_raw.reset_index()

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

results = pd.DataFrame()
ix = 0 # index for columns in results
for i in range(20):
    #X = data_sterile[features]
    labels = ((data_sterile['t_serum_conc_percent']) / 10).astype('int')
    groups = data_sterile['file']
    gkf = StratifiedGroupKFold(n_splits=4, shuffle=True)
    for train_idxs, test_idxs in gkf.split(data_sterile[features], labels, groups):
        X = data_sterile[features].loc[train_idxs]
        X_test = data_sterile[features].loc[test_idxs]
        y = labels.loc[train_idxs]
        y_test = labels.loc[test_idxs]
        scaler = StandardScaler()
        scaler.fit(X)
        X_norm = scaler.transform(X)
        X_test_norm = scaler.transform(X_test)
        model = models.Sequential()
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(X_norm,
                            y,
                            epochs=500,
                            # batch_size=50,
                            validation_data=(X_test_norm, y_test),
                            verbose=0,
                            )
        results["acc" + str(ix)] = history.history['accuracy']
        results["val_acc" + str(ix)] = history.history['val_accuracy']
        ix += 1
        # print(1-y.sum()/len(y))
        # print(1 - y_test.sum() / len(y_test))
        print(ix)
dnn_results = results.iloc[:, range(1, len(results.columns), 2)]
dnn_results['mean'] = dnn_results.mean(axis=1)
plt.plot(dnn_results['mean'])
plt.show()
