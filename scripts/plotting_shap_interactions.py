import pandas as pd
from matplotlib import pyplot as plt, rcParams
import seaborn as sns
import numpy as np
import shap

from matplotlib import pyplot as plt, rcParams

rcParams['figure.dpi'] = 300
rcParams.update({'figure.autolayout': True})

gbc_shap = pd.read_csv('data/20230308_75f10d8c_shap_averaged_GBC.csv')
shaps = gbc_shap.iloc[:, 20:]
X = gbc_shap.iloc[:, :20]


def approximate_interactions(index, shap_values, X, feature_names=None):
    # that's from shap utils https://github.com/slundberg/shap/blob/master/shap/utils/_general.py

    """ Order other features by how much interaction they seem to have with the feature at the given index.
    This just bins the SHAP values for a feature along that feature's value. For true Shapley interaction
    index values for SHAP see the interaction_contribs option implemented in XGBoost.
    """
    # convert from DataFrames if we got any
    if str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = X.columns
        X = X.values
    index = convert_name(index, shap_values, feature_names)
    if X.shape[0] > 10000:
        a = np.arange(X.shape[0])
        np.random.shuffle(a)
        inds = a[:10000]
    else:
        inds = np.arange(X.shape[0])
    x = X[inds, index]
    srt = np.argsort(x)
    shap_ref = shap_values[inds, index]
    shap_ref = shap_ref[srt]
    inc = max(min(int(len(x) / 10.0), 50), 1)
    interactions = []
    for i in range(X.shape[1]):
        encoded_val_other = encode_array_if_needed(X[inds, i][srt], dtype=np.float)
        val_other = encoded_val_other
        v = 0.0
        if not (i == index or np.sum(np.abs(val_other)) < 1e-8):
            for j in range(0, len(x), inc):
                if np.std(val_other[j:j + inc]) > 0 and np.std(shap_ref[j:j + inc]) > 0:
                    v += abs(np.corrcoef(shap_ref[j:j + inc], val_other[j:j + inc])[0, 1])
        val_v = v
        val_other = np.isnan(encoded_val_other)
        v = 0.0
        if not (i == index or np.sum(np.abs(val_other)) < 1e-8):
            for j in range(0, len(x), inc):
                if np.std(val_other[j:j + inc]) > 0 and np.std(shap_ref[j:j + inc]) > 0:
                    v += abs(np.corrcoef(shap_ref[j:j + inc], val_other[j:j + inc])[0, 1])
        nan_v = v
        interactions.append(max(val_v, nan_v))
    return np.argsort(-np.abs(interactions)), np.abs(interactions)


def convert_name(ind, shap_values, input_names):
    # that's from shap utils https://github.com/slundberg/shap/blob/master/shap/utils/_general.py
    if type(ind) == str:
        nzinds = np.where(np.array(input_names) == ind)[0]
        if len(nzinds) == 0:
            # we allow rank based indexing using the format "rank(int)"
            if ind.startswith("rank("):
                return np.argsort(-np.abs(shap_values).mean(0))[int(ind[5:-1])]
            # we allow the sum of all the SHAP values to be specified with "sum()"
            # assuming here that the calling method can deal with this case
            elif ind == "sum()":
                return "sum()"
            else:
                raise ValueError("Could not find feature named: " + ind)
        else:
            return nzinds[0]
    else:
        return ind


def encode_array_if_needed(arr, dtype=np.float64):
    # that's from shap utils https://github.com/slundberg/shap/blob/master/shap/utils/_general.py
    try:
        return arr.astype(dtype)
    except ValueError:
        unique_values = np.unique(arr)
        encoding_dict = {string: index for index, string in enumerate(unique_values)}
        encoded_array = np.array([encoding_dict[string] for string in arr], dtype=dtype)
        return encoded_array


# approximate_interactions(index = '19r_MD', shap_values=shaps.to_numpy(), X=X.to_numpy(),
#                         feature_names=X.columns.str[4:])
inters_rank = pd.DataFrame()
inters = pd.DataFrame()

# number here represents feature index in feature_list, row index - rank in importance
for i in X.columns.str[4:]:
    intrs = approximate_interactions(index=i, shap_values=shaps.to_numpy(), X=X.to_numpy(),
                                     feature_names=X.columns.str[4:])

    inters[i] = intrs[1]
    inters_rank[i] = intrs[0]

inters.index = X.columns.str[4:]

sns.heatmap(inters, annot=False, square=True, cbar=False, cmap='coolwarm')
plt.show()
plt.close()

''' Bump plot for importance gbc/mlp '''

feature_ranks = pd.DataFrame()

# baseline accs to add to bump chart
# note, acc of 1 level decision tree is used as feature importance.
# those are calculated in plotting_baselines.py
# max acc is higher than baseline, as there is no free feature selection, so it's ok

tree_accs = pd.read_csv('data/20230321_7a46f7a9_acc_1lvlTREE.csv', index_col='index')
tree_accs = tree_accs['base_sf_rank']
tree_accs.loc[['ifFast', 'ifCentr', 'out2sd', 'out3sd']] = np.nan  # no meaning for thresholding
feature_ranks['tree_ranks'] = tree_accs

# gbc mean abs shaps and rank

gbc_shap = pd.read_csv('data/20230321_7a46f7a9_shap_averaged_GBC.csv')
shaps = gbc_shap.iloc[:, 20:]
X = gbc_shap.iloc[:, :20]
m_shaps = shaps.abs().mean(0)
m_shaps.index = shaps.columns.str[9:]
feature_ranks['gbc_m_shaps'] = m_shaps
feature_ranks['gbc_ranks'] = np.argsort(np.argsort(m_shaps))

# mlp mean abs shaps and rank

mlp_shap = pd.read_csv('data/20230317_7a46f7a9_shap_averaged_MLP.csv')
nn_shaps = mlp_shap.iloc[:, 20:]
nn_X = mlp_shap.iloc[:, :20]
nn_m_shaps = nn_shaps.abs().mean(0)
nn_m_shaps.index = nn_shaps.columns.str[9:]
feature_ranks['mlp_m_shaps'] = nn_m_shaps
feature_ranks['mlp_ranks'] = np.argsort(np.argsort(nn_m_shaps))

# plot
to_plot = feature_ranks[['tree_ranks', 'gbc_ranks', 'mlp_ranks']]
long_feature_ranks = pd.melt(to_plot, ignore_index=False).reset_index()
plt.rcParams["figure.figsize"] = (2.5, 5)

fig, ax = plt.subplots()
ax.axis('off')

for i in shaps.columns.str[9:]:
    ax.plot(long_feature_ranks.loc[long_feature_ranks['index'] == i]['variable'],
            long_feature_ranks.loc[long_feature_ranks['index'] == i]['value'],
            "o-",
            markerfacecolor='white',
            linewidth=3,
            )

plt.show()
plt.close()

feature_ranks.reset_index(inplace=True)
