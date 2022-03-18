'''
This is to try unsupervised learning (PCA, UMAP etc) for feature analysis, dimensionality
reduction, visualization, importance estimation.
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns

''' 
read and scale the data 
'''

data = pd.read_csv('data/b212a935_Chr1_data_sterile.csv')

fset1 = [
    'f_mean_diff_xy_micron', 'f_max_diff_xy_micron', 'f_var_diff_xy_micron',
    'f_area_micron', 'f_perimeter_au_norm', 'f_min_dist_micron',
    'f_var_dist_micron', 'f_Rvar_diff_xy_micron', 'f_Rvar_dist_micron',
    'f_total_displacement', 'f_persistence',
    'f_min_dist_range', 'f_total_min_dist',
    'f_slope_min_dist_micron', 'f_slope_area_micron',
    'f_slope_perimeter_au_norm'
]
# no masks

fset2 = [
    'f_mean_diff_xy_micron', 'f_area_micron', 'f_perimeter_au_norm', 'f_min_dist_micron'
]




X = data[fset1]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

'''
Principal Component Analysis (PCA) as in Muller and Guido
'''

pca = PCA()
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

to_plot = pd.DataFrame(X_pca).add_prefix('PC')
to_plot['serum']=data['t_serum_conc_percent']

# sns.relplot(data=to_plot, x='PC0', y='PC1', hue='serum')
sns.displot(data=to_plot, x='PC0', y='PC1', hue='serum', kind="kde")
plt.show()


'''
add principal components to input DataFrame
'''

data_pca = data.join(pd.DataFrame(X_pca).add_prefix('PC'))




