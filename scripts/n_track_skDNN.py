"""
This is to try keras models in sklearn environment (i.e. to use GroupKFold or grid search)
as here
https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
"""

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


''' 
data preprocessing 
'''

# X = data_raw[data_raw.columns[1:]]
X = data_raw[data_raw.columns[1:21]]  # these are engineered features only
# X = data_raw[data_raw.columns[37:]] #  raw features only
X_norm = X / X.max(axis=0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = ((data_raw['t_serum_conc_percent']) / 10).astype('int')

''' 
building keras model

technically it is possible to gridsearch through hidden layers number as hyperparameter
https://stackoverflow.com/questions/47788799/grid-search-the-number-of-hidden-layers-with-keras

'''


def create_model():
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
    return model


''' 
use model with sklearn cross-val
'''
for k in range(5, 600, 5):  # takes an eternity, needed to check for accuracy 'degradation' due to overfitting
    model = KerasClassifier(build_fn=create_model, epochs=k, verbose=0)
    gkf = StratifiedGroupKFold(n_splits=4, shuffle=True)
    results = pd.DataFrame(columns=['spl1', 'spl2', 'spl3', 'spl4'])
    for i in range(10):  # repeated CV, since one iteration gives too unstable results
        scores = cross_val_score(model, X_norm, y, cv=gkf, groups=data_raw.reset_index()['file'])
        results = results.append(pd.Series(scores, index=results.columns), ignore_index=True)

    print(str(k) + ' epochs: ' + str(results.mean().mean()))

''' 
(one time version) use model with sklearn cross-val 
'''

model = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)
gkf = StratifiedGroupKFold(n_splits=4, shuffle=True)
results = pd.DataFrame(columns=['spl1', 'spl2', 'spl3', 'spl4'])
for i in range(10):  # repeated CV, since one iteration gives too unstable results
    scores = cross_val_score(model, X_norm, y, cv=gkf, groups=data_raw.reset_index()['file'])
    results = results.append(pd.Series(scores, index=results.columns), ignore_index=True)

print(results.mean().mean())

'''
pipeline version, to avoid data leakage from scaling before splitting 
'''

model_pipe = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)
DNN_pipeline = Pipeline(
    steps=[('scaler', StandardScaler()),
           ('DNN', model_pipe)]
)
results = pd.DataFrame(columns=['spl1', 'spl2', 'spl3', 'spl4'])
for i in range(30):  # repeated CV, since one iteration gives too unstable results
    print('iter ' + str(i) + ' start, time:', datetime.now().strftime("%H:%M:%S"))
    scores = cross_val_score(DNN_pipeline, X, y, cv=gkf, groups=data_raw.reset_index()['file'])
    results = results.append(pd.Series(scores, index=results.columns), ignore_index=True)

print(results.mean().mean())

'''
SHAP explanation of DNN
'''
# repeated 10 times the code below returns average 0.640484

pred_list = []
pred_proba_list = []
shap_vs_list = []
sX_test_list = []
sy_test_list = []
s_id_list = []

for strain, stest in gkf.split(X_scaled, y, data_raw.reset_index()['file']):
    test_data = data_raw.reset_index().iloc[stest, :]  # kept here for s_id_list
    sX = pd.DataFrame(X_scaled).iloc[strain, :]
    sy = y.iloc[strain]
    sX_test = pd.DataFrame(X_scaled).iloc[stest, :]
    sy_test = y.iloc[stest]

    sX_test_list.append(sX_test)
    sy_test_list.append(sy_test)
    s_id_list.append(test_data[['file', 'particle']])

    model.fit(sX, sy)

    pred = model.predict(sX_test)
    pred_list.append(pred)
    pred_proba = model.predict_proba(sX_test)
    pred_proba_list.append(pred_proba)

    explainer = shap.KernelExplainer(model, sX_test)
    # shap_values = explainer.shap_values(sX_test)
    # shap_vs_list.append(shap_values)

    # shap.summary_plot(shap_values, sX_test, sort=False, color_bar=False, plot_size=(10,10))

all_sX_test = pd.concat(sX_test_list)
all_sy_test = pd.concat(sy_test_list)
# all_splits_shap = np.concatenate(shap_vs_list)
all_pred = np.concatenate(pred_list)
all_pred_proba = np.concatenate(pred_proba_list)
all_s_id = pd.concat(s_id_list)

list_to_concat = [all_sX_test.reset_index(),
                  all_sy_test.reset_index(),
                  # df_all_splits_shap,
                  pd.DataFrame(all_pred, columns=['predicted']),
                  pd.DataFrame(all_pred_proba).add_prefix('proba_'),
                  all_s_id]

df_all = pd.concat(list_to_concat, axis=1)
df_all['correct'] = (df_all['t_serum_conc_percent'] == df_all['predicted'])
print((np.sum(df_all['correct'])) / (len(df_all)))

'''
SHAP explanation of DNN attempt
Weird unstructured block, which somehow gives realistic val_accuracy in splits, and even returns some sort of 
averaged SHAP values for each feature. The rest doesn't work, no swarm plot, no aggregation for splits etc.
'''

gkf = StratifiedGroupKFold(n_splits=4, shuffle=True)


pred_list = []
pred_proba_list = []
shap_vs_list = []
sX_test_list = []
sy_test_list = []
s_id_list = []

for strain, stest in gkf.split(X_scaled, y, data_raw.reset_index()['file']):
    #test_data = data_raw.reset_index().iloc[stest, :]  # kept here for s_id_list
    sX = pd.DataFrame(X_scaled, columns=X.columns).iloc[strain, :]
    sy = y.iloc[strain]
    sX_test = pd.DataFrame(X_scaled, columns=X.columns).iloc[stest, :]
    sy_test = y.iloc[stest]

    sX_test_list.append(sX_test)
    sy_test_list.append(sy_test)
    #s_id_list.append(test_data[['file', 'particle']])

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


    model.fit(sX, sy, epochs=100, validation_data=(sX_test, sy_test))

    pred = model.predict(sX_test)
    pred_list.append(pred)
    # pred_proba = model.predict_proba(sX_test)
    # pred_proba_list.append(pred_proba)

    explainer = shap.KernelExplainer(model, sX_test)
    shap_values = explainer.shap_values(sX_test)
    shap_vs_list.append(shap_values)

    shap.summary_plot(shap_values, sX_test, sort=False, color_bar=False, plot_size=(25,10))
    #shap.plots.beeswarm(shap_values, max_display=20)

all_sX_test = pd.concat(sX_test_list)
all_sy_test = pd.concat(sy_test_list)
all_splits_shap = np.concatenate(shap_vs_list)
all_pred = np.concatenate(pred_list)
#all_pred_proba = np.concatenate(pred_proba_list)
#all_s_id = pd.concat(s_id_list)


#plt.title('aggregated')
shap.summary_plot(all_splits_shap, all_sX_test, sort=False, color_bar=False, plot_size=(25, 10))

