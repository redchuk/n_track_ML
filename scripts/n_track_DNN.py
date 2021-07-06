import pandas as pd
import numpy as np

''' read the data '''
data = pd.read_csv('scripts/data_chromatin_live.csv')

''' 
add features 

In a resulting table targets have 't' in a column name, while features to be used in training start with 'f'.
'''
data_agg = data.groupby(['file', 'particle']).agg(t_guide=('guide', 'first'),
                                                  t_time=('time', 'first'),
                                                  t_serum_conc_percent=('serum_conc_percent', 'first'),

                                                  f_mean_diff_xy_micron=('diff_xy_micron', 'mean'),
                                                  # average displacement
                                                  f_max_diff_xy_micron=('diff_xy_micron', 'max'),  # maximal displacement
                                                  f_sum_diff_xy_micron=('diff_xy_micron', 'sum'),
                                                  # total trajectory length

                                                  sum_diff_x_micron=('diff_x_micron', 'sum'),
                                                  sum_diff_y_micron=('diff_y_micron', 'sum'),

                                                  )

data_agg['f_total_displacement'] = np.sqrt((data_agg['sum_diff_x_micron'])**2 + (data_agg['sum_diff_y_micron'])**2)
# distance from first to last coordinate
data_agg['f_persistence'] = data_agg['f_total_displacement']/data_agg['f_sum_diff_xy_micron']
# shows how directional the movement is

data_agg['file_mean_diff_xy_micron'] = data_agg.groupby('file')['f_mean_diff_xy_micron'].transform(np.max)
data_agg['f_fastest_mask'] = np.where((data_agg['f_mean_diff_xy_micron']==data_agg['file_mean_diff_xy_micron']), 1, 0)
# the fastest (or the only available) dot in the nucleus is 1, the rest is 0


''' stratified cross-val K fold '''
