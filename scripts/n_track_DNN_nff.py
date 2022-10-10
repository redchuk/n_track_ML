'''
unlike in n_track_DNN, here the files with <30 frames are dropped
somehow this gives +2% acc
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

'''
from keras.wrappers.scikit_learn import KerasClassifier
from keras import models
from keras import layers
'''

import tensorflow as tf

# https://stackoverflow.com/questions/66814523/shap-deepexplainer-with-tensorflow-2-4-error
tf.compat.v1.disable_v2_behavior()
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout

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

ix = 0  # index for columns in results
shap_repeats = pd.DataFrame()

for i in range(2):
    # X = data_sterile[features]

    shap_vs_list = []
    sX_test_list = []
    s_id_list = []


    results = pd.DataFrame()

    labels = ((data_sterile['t_serum_conc_percent']) / 10).astype('int')
    groups = data_sterile['file']
    gkf = StratifiedGroupKFold(n_splits=4, shuffle=True)
    idx_file_particle = data_sterile[['file', 'particle']]

    for train_idxs, test_idxs in gkf.split(data_sterile[features], labels, groups):
        X = data_sterile[features].loc[train_idxs]
        X_test = data_sterile[features].loc[test_idxs]
        y = labels.loc[train_idxs]
        y_test = labels.loc[test_idxs]
        scaler = StandardScaler()
        scaler.fit(X)
        X_norm = scaler.transform(X)
        X_test_norm = scaler.transform(X_test)

        sX_test_list.append(pd.DataFrame(X_test_norm, columns=X_test.columns))  # X_test?
        s_id_list.append(idx_file_particle.loc[test_idxs])

        input_layer = layers.Input(shape=(20,))
        layer1 = layers.Dense(256, activation='relu')(input_layer)
        layer2 = layers.Dense(256, activation='relu')(layer1)
        layer3 = layers.Dense(256, activation='relu')(layer2)
        layer4 = layers.Dense(256, activation='relu')(layer3)
        layer5 = layers.Dense(256, activation='relu')(layer4)
        layer6 = layers.Dense(256, activation='relu')(layer5)
        layer7 = layers.Dense(256, activation='relu')(layer6)
        output_layer = layers.Dense(1, activation="sigmoid")(layer7)
        model = models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(X_norm,
                            y,
                            epochs=200,
                            # batch_size=50,
                            validation_data=(X_test_norm, y_test),
                            verbose=0,
                            )

        results["acc" + str(ix)] = history.history['acc']
        results["val_acc" + str(ix)] = history.history['val_acc']
        ix += 1

        explainer = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), X_norm)
        shap_values = explainer.shap_values(X_test_norm)
        shap_vs_list.append(shap_values[0])

        shap.summary_plot(shap_values[0],
                          X_test_norm,
                          feature_names=X.columns,
                          sort=False,
                          color_bar=False,
                          plot_size=(10, 10),
                          )

        # print(1-y.sum()/len(y))
        # print(1 - y_test.sum() / len(y_test))
        print(ix)

    all_sX_test = pd.concat(sX_test_list)
    all_splits_shap = np.concatenate(shap_vs_list)
    all_s_id = pd.concat(s_id_list)

    plt.title('aggregated')
    shap.summary_plot(all_splits_shap,
                      all_sX_test,
                      feature_names=X.columns,
                      sort=False,
                      color_bar=False,
                      plot_size=(10, 10),
                      )

    df_all_splits_shap = pd.DataFrame(all_splits_shap, columns=all_sX_test.columns).add_prefix('shap_')

    list_to_concat = [all_sX_test.reset_index(),
                      df_all_splits_shap.reset_index(),
                      all_s_id.reset_index()]

    df_all = None
    df_all = pd.concat(list_to_concat, axis=1) \
        .drop('index', axis=1).set_index(['file', 'particle']).add_prefix(str(i) + 'r_')

    if shap_repeats.empty:
        shap_repeats = df_all
    else:
        shap_repeats = shap_repeats.join(df_all)

dnn_results = results.iloc[:, range(1, len(results.columns), 2)]
dnn_results['mean'] = dnn_results.mean(axis=1)
plt.plot(dnn_results['mean'])
plt.show()

# todo: averaging SHAP
