"""
This is to try keras models in sklearn environment (i.e. to use GroupKFold or grid search)
as here
https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from keras import models
from keras import layers
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score

''' 
read the data 

'''
data = pd.read_csv('scripts/a286935_data_chromatin_live.csv')
data = data[~data["comment"].isin(["stress_control"])]
data = data[~data["comment"].isin(["H2B"])]
data = data[data["guide"].str.contains('1398') | data["guide"].str.contains('1514')]
data = data[data["time"] < 40]
data.set_index(['file', 'particle'], inplace=True)

# original data before aggregation

data_sterile = pd.read_csv('scripts/data_sterile_PCA_92ba95d.csv')
data_sterile.set_index(['file', 'particle'], inplace=True)

# data after aggregation and feature engineering

# broadcast the engineered features to original time series

# create feature sets, 'original', 'all'






''' 
data preprocessing 
'''

X = data_sterile[features]
X_norm = X / X.max(axis=0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = ((data_sterile['t_serum_conc_percent']) / 10).astype('int')

''' 
building keras model
'''


def create_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

''' 
use model with sklearn cross-val
'''


model = KerasClassifier(build_fn=create_model, epochs=500, verbose=0)
# batch size?
gkf = GroupKFold(n_splits=4)
results = pd.DataFrame(columns=['spl1', 'spl2', 'spl3', 'spl4'])
for i in range(10):
    scores = cross_val_score(model, X_norm, y, cv=gkf, groups=data_sterile['file'])
    results = results.append(pd.Series(scores, index=results.columns), ignore_index=True)

# this gives accuracy 0.6223815754055977