import pandas as pd
import dabest
import seaborn as sns
from matplotlib import pyplot as plt, rcParams
from numpy import mean, median
import math
from scipy import stats
import shap

# mpl colors https://matplotlib.org/stable/gallery/color/named_colors.html

rcParams['figure.dpi'] = 300
rcParams.update({'figure.autolayout': True})

data_to_plot = pd.read_csv('data/data_sterile_f67592f.csv')
# data_to_plot = data_to_plot[data_to_plot['f_outliers2SD_diff_xy'] > 1] # outliers only

"""
Plotting the dynamic features using Dabest
"""


def gen_set(data, parameter='', chr_name='', guides=''):
    """

    :param data: df
    :param parameter: feature to plot, str
    :param chr_name: chr name for labeling, str
    :param guides: guides to aggregate, str, use upright slash for list of guides
    :return: tuple for Dabest plotting, format:
                        idx=(("Control 1", "Test 1",),
                             ("Control 2", "Test 2")
                             ))

    """

    data = data[data["t_guide"].str.contains(guides, regex=True)]  # .dropna()
    before = pd.Series(data=data[data["t_time"] == 0][parameter], name=chr_name + ', 10%')
    after = pd.Series(data=data[(data["t_time"] < 40) & (data["t_time"] > 0)][parameter], name=chr_name + ', 0.3%')
    frame = pd.concat([before, after], axis=1)
    return frame


'''
Guides:
'pl_1521_chr10', 'pl_1522_chr10', 'pl_1404_chr13', 'pl_1406_chrx',
'pl_1398_chr1', 'pl_1532_chr18', 'pl_1362_telo', 'pl_1514_chr1',
'pl_1403_chr13'

'''

feature = 'f_area_micron'
hue = 'f_outliers2SD_diff_xy'

chr1 = gen_set(data_to_plot, chr_name='chr1', parameter=feature, guides='1398|1514')
chr10 = gen_set(data_to_plot, chr_name='chr10', parameter=feature, guides='1521|1522')
chr13 = gen_set(data_to_plot, chr_name='chr13', parameter=feature, guides='1403|1404')
chrX = gen_set(data_to_plot, chr_name='chrX', parameter=feature, guides='1406')
telo = gen_set(data_to_plot, chr_name='telo', parameter=feature, guides='1362')

all_chr = pd.concat([chr1,
                     # chr10,
                     # chr13,
                     # chrX,
                     # telo,
                     data_to_plot[hue]], axis=1)
# all_chr.columns

multi_2group = dabest.load(all_chr, idx=(('chr1, 10%', 'chr1, 0.3%'),
                                         # ('chr10, 10%', 'chr10, 0.3%'),
                                         # ('chr13, 10%', 'chr13, 0.3%'),
                                         # ('chrX, 10%', 'chrX, 0.3%'),
                                         # ('telo, 10%', 'telo, 0.3%'),
                                         ))
# CI is 95% by default in dabest
print(multi_2group.mean_diff)

multi_2group.mean_diff.plot(raw_marker_size=3,
                            es_marker_size=6,
                            swarm_label=feature,
                            # color_col=hue,
                            swarm_desat=1,
                            custom_palette='Paired',
                            # swarm_ylim=(-0.05, 0.05),
                            )

plt.show()
plt.close()

"""
Plotting the correlations in serum and in starvation, to check for homolog_chr-specific behaviour
Chromosome 1 only

"""

chr1 = data_to_plot[data_to_plot["t_guide"].str.contains('1398|1514', regex=True)]
chr1.set_index(['file', 'particle'], inplace=True)

# features = ['f_min_dist_micron', 'f_min_dist_range', 'f_total_min_dist',
# 'f_slope_min_dist_micron', 'f_mean_diff_xy_micron', 'f_max_diff_xy_micron',
# 'f_persistence', 'f_total_displacement']

features = ['f_total_min_dist', 'f_mean_diff_xy_micron', 'f_total_displacement']

# fast / slow

fast = chr1[chr1['f_fastest_mask'] == 1].reset_index(level=1).add_prefix('f_')
slow = chr1[chr1['f_fastest_mask'] == 0].reset_index(level=1).add_prefix('s_')
slow = slow[~slow.index.duplicated(keep='first')]  # in some (rare) cases the dot retained is not the slowest one
speed = pd.concat([fast, slow], axis=1).dropna()

speed = speed[speed["f_t_time"] < 40]
for i in features:
    fig = sns.lmplot(x='s_' + i, y='f_' + i, data=speed)
    fig.savefig('C:/Users/redchuk/python/temp/temp_n_track_RF/summry20210923/r_sq/' + 'fs_' + str(i) + '.png')
    plt.close()

# central / peripheral

central = chr1[chr1['f_most_central_mask'] == 1].reset_index(level=1).add_prefix('c_')
peripheral = chr1[chr1['f_most_central_mask'] == 0].reset_index(level=1).add_prefix('p_')
peripheral = peripheral[~peripheral.index.duplicated(keep='first')]
c_p = pd.concat([central, peripheral], axis=1).dropna()

# Pearson coefficient works with a linear relationship whereas the Spearman works with all monotonic relationships

c_p = c_p[c_p['c_t_time'] < 40]
for i in features:
    spear, spval = stats.spearmanr(a=c_p['p_' + i], b=c_p['c_' + i])

    fig = sns.lmplot(x='p_' + i, y='c_' + i, data=c_p, aspect=1)
    # fig.set(xlim=(0, 1.1))
    # fig.set(ylim=(0, 1.1))
    # .set_axis_labels
    plt.title('spear=' + str(spear) + ', p=' + str(spval))
    fig.savefig('C:/Users/redchuk/python/temp/temp_n_track_RF/summry20210923/r_sq/' + 'cp_' + str(i) + '.png')
    plt.close()

"""
Plotting fixed cells data (obtained by Olli Natri)

"""

fixed_data = pd.read_csv('C:/Users/redchuk/python/temp/temp_n_track_RF/fixed/fixed_data_combined_20200730.csv')
fixed_data['chromosome'] = fixed_data['guide'].str[:4]
fixed_data = fixed_data[~fixed_data['date'].isin(['23.10.2019'])]
# ['23.10.2019', '14.1.2020', '14.5.2020', '23.7.2020']

violin = sns.catplot(
    data=fixed_data,
    kind='violin',
    x='chromosome',
    y='min_dist_micron',
    hue='time',
    split=True,
    inner='quartiles',
)
violin.set(ylim=(0, None))

box = sns.catplot(
    data=fixed_data,
    kind='bar',
    x='chromosome',
    y='min_dist_micron',
    hue='time',
    estimator=mean,
    ci=None
)

"""
Bar graphs for shells, fixed cells, chr 1 (Olli's 20200114)

"""

serum = [0, 2, 10, 24, 32]
starvation = [0, 6, 14, 27, 31]
bars = pd.DataFrame({'10%': serum, '0.3%': starvation})
bars_norm = bars / bars.sum()

"""
Morphology plotting

"""

# feature = 'f_perimeter_au_norm'
feature = 'circularity'

morph_plot = pd.read_csv('data/data_sterile_f67592f.csv')
morph_plot = morph_plot.groupby(['file']).agg({'f_area_micron': 'first',
                                               'f_perimeter_au_norm': 'first',
                                               't_time': 'first',
                                               't_guide': 'first',
                                               't_serum_conc_percent': 'first',
                                               })

morph_plot['circularity'] = 4 * math.pi * morph_plot['f_area_micron'] / morph_plot['f_perimeter_au_norm'] ** 2
morph_plot['circularity'] = morph_plot['circularity'] / morph_plot['circularity'].max()

# Circularity = (4 * pi * Area) / Perimeter^2
# since we have perimeter in au, we have to divide by max to get something circularity-ish

chr1_telo = gen_set(morph_plot, chr_name='morph', parameter=feature, guides='1398|1514|1362')

two_group = dabest.load(chr1_telo, idx=(('morph, 10%', 'morph, 0.3%'),
                                        ))
# CI is 95% by default in dabest

two_group.mean_diff.plot(raw_marker_size=4,
                         es_marker_size=6,
                         swarm_label=feature,
                         # color_col=hue,
                         swarm_desat=1,
                         custom_palette='Paired',
                         swarm_ylim=(0, 1),
                         )

"""
MLP plotting
"""

# accuracy/epoch

validation_profiles = pd.read_csv('data/20230317_7a46f7a9_validation_profiles_MLP.csv')
mlp_vals_long = pd.melt(validation_profiles, ignore_index=False)
mlp_vals_long.reset_index(inplace=True)
rcParams.update({'figure.autolayout': True})
sns.set_style(style='ticks')
p = sns.relplot(
    data=mlp_vals_long, kind="line", x='index', y='value')
p.set(ylim=(0.57, 0.66))
plt.gcf().set_size_inches(4, 3.5)
plt.show()

# accuracy/cv repeats, for 200th epoch

list_acc = validation_profiles.iloc[-1]
acc_df = pd.DataFrame()
for i in reversed(range(3, len(list_acc), 4)):
    acc_df[i] = list_acc[:i]

acc_df_long = acc_df.melt()
rcParams.update({'figure.autolayout': True})
sns.set_style(style='ticks')
sns.relplot(data=acc_df_long.dropna(), kind='line', x='variable', y='value')
p.set(ylim=(0.57, 0.66))  # doesn't work here
plt.gcf().set_size_inches(4, 3.5)
plt.show()

# MLP SHAP

mlp_shap = pd.read_csv('data/20230317_7a46f7a9_shap_averaged_MLP.csv')


def sh_plot(shap_values, feature_values, feature_names):
    shap.summary_plot(shap_values,
                      feature_values,
                      feature_names=feature_names,
                      sort=False,
                      color_bar=False,
                      plot_size=(10, 10),
                      show=False,
                      )


fnames = mlp_shap.iloc[:, :20].columns.str[4:]
sh_plot(mlp_shap.iloc[:, 20:].to_numpy(), mlp_shap.iloc[:, :20].to_numpy(), fnames)
rcParams['figure.dpi'] = 200
plt.title('MLP, 20 CV repeats')
plt.gcf().set_size_inches(6, 6)
plt.show()

"""
GBC plotting
"""

# accuracy/cv repeats

import numpy as np

pivots_df = pd.read_csv('data/20230317_7a46f7a9_pivots_GBC.csv')
list_acc = list(pivots_df.loc[0.01, 5].reset_index(drop=True))
acc_df = pd.DataFrame()
for i in reversed(range(2, len(list_acc))):
    #    print(i)
    #   print(list_acc[:i])
    acc_df[i] = pd.Series(list_acc[:i])
acc_df_long = acc_df.melt()
rcParams.update({'figure.autolayout': True})
sns.set_style(style='ticks')
p = sns.relplot(data=acc_df_long.dropna(), kind='line', x='variable', y='value', color='steelblue')
p.set(ylim=(0.54, 0.64))
p.set(xlim=(1, 20))
p.set(xticks=np.arange(1, 21, 2))
plt.gcf().set_size_inches(4, 3.5)
plt.show()

# GBC SHAP

path = 'data/20230321_7a46f7a9_shap_averaged_GBC.csv'
gbc_shap = pd.read_csv(path)


def sh_plot(shap_values, feature_values, feature_names):
    shap.summary_plot(shap_values,
                      feature_values,
                      feature_names=feature_names,
                      sort=False,
                      color_bar=False,
                      plot_size=(10, 10),
                      show=False,
                      )


fnames = gbc_shap.iloc[:, :20].columns.str[4:]
sh_plot(gbc_shap.iloc[:, 20:].to_numpy(), gbc_shap.iloc[:, :20].to_numpy(), fnames)
rcParams['figure.dpi'] = 200
plt.title('GBC, 20 CV repeats')
plt.gcf().set_size_inches(6, 6)
plt.show()

# GBC SHAP dependence  /  can be used for MLP

path = 'data/20230321_7a46f7a9_shap_averaged_GBC.csv'
gbc_shap = pd.read_csv(path)

# list1 = [('MD', 'MA'), ('MDist', 'MA'), ('MA', 'MD'), ('Pers', 'MA')]
# list1 = [('MD', 'Pers'), ('MDist', 'sDist'), ('MA', 'rVarD'), ('Pers', 'out3sd')]
list1 = [('sA', 'ifCentr'), ('sDist', 'ifCentr'), ('TDist', 'ifCentr'), ('sDist', 'ifCentr')]  # top engineered by MLP

fnames = gbc_shap.iloc[:, :20].columns.str[4:]


def sh_dep_plot(feature, shap_values, feature_values, fnames, color):
    shap.dependence_plot(feature,
                         shap_values,
                         feature_values,
                         fnames,
                         interaction_index=color,
                         show=False,
                         alpha=0.7,
                         )


for i in list1:
    sh_dep_plot(i[0], gbc_shap.iloc[:, 20:].to_numpy(), gbc_shap.iloc[:, :20].to_numpy(), fnames, i[1])
    rcParams['figure.dpi'] = 200
    plt.title(path.split('/')[1].split('.')[0])
    plt.gcf().set_size_inches(6, 4)
    plt.show()
    plt.close()

"""
Feature correlation colored by SHAP
"""

'''
['MD', 'MaxD', 'VarD', 'MA', 'MP', 'MDist', 'VarDist', 'rVarD',
 'rVarDist', 'TD', 'Pers', 'ifFast', 'DistR', 'TDist', 'ifCentr',
 'sDist', 'sA', 'sP', 'out2sd', 'out3sd']
'''

path = 'data/20230317_7a46f7a9_shap_averaged_MLP.csv'
x_n_shaps = pd.read_csv(path)
x_n_shaps.columns = x_n_shaps.columns.str[4:]

sns.relplot(data=x_n_shaps,
            x='MD', y='MDist',
            # hue='MA',
            # alpha=0.9,
            # palette='Spectral',
            # size='MA',
            # sizes=(1, 100),
            )
# plt.ylim(-0.9,)
# plt.xlim(110, 790)
plt.show()
plt.close()

"""
post-SHAP plotting
feature correlation 
"""

data_to_plot = pd.read_csv('data/data_sterile_f67592f.csv')

g_chr1 = '1398|1514'
g_chr10 = '1521|1522'
g_chr13 = '1403|1404'
g_chrX = '1406'
g_telo = '1362'

data_to_plot = data_to_plot[data_to_plot['t_guide'].str.contains(g_chr1, regex=True)]  # .dropna()

canonical_fnames = {'diff_xy_micron': 'D',  # Locus displacement, microns
                    'area_micron': 'A',  # Nuclear area, square microns
                    'perimeter_au_norm': 'P',  # Nuclear perimeter, arbitrary units
                    'min_dist_micron': 'Dist',  # Minimal distance from locus to nuclear rim, microns
                    }

# next
data_to_plot.rename(columns=canonical_fnames, inplace=True)

# next
'''
sns.relplot(data=data_to_plot,
            x='MD', y='MDist',
            #hue='MA',
            #alpha=0.9,
            #palette='Spectral',
            #size='MA',
            #sizes=(1, 100),
            )
#plt.ylim(-0.9,)
#plt.xlim(110, 790)
plt.show()
plt.close()

'''
