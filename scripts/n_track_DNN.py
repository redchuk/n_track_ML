import pandas as pd
import numpy as np
#from scipy.stats import linregress

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
                                                  f_max_diff_xy_micron=('diff_xy_micron', 'max'),
                                                  # maximal displacement
                                                  f_sum_diff_xy_micron=('diff_xy_micron', 'sum'),
                                                  # total trajectory length
                                                  f_var_diff_xy_micron=('diff_xy_micron', 'var'),
                                                  # variance in displacements

                                                  sum_diff_x_micron=('diff_x_micron', 'sum'),
                                                  sum_diff_y_micron=('diff_y_micron', 'sum'),

                                                  f_area_micron=('area_micron', 'mean'),
                                                  f_perimeter_au_norm=('perimeter_au_norm', 'mean'),
                                                  # morphology

                                                  f_min_dist_micron=('min_dist_micron', 'mean'),
                                                  min_min_dist_micron=('min_dist_micron', 'min'),
                                                  max_min_dist_micron=('min_dist_micron', 'max'),
                                                  beg_min_dist_micron=('min_dist_micron', 'first'),
                                                  end_min_dist_micron=('min_dist_micron', 'last'),
                                                  )

data_agg['f_total_displacement'] = np.sqrt((data_agg['sum_diff_x_micron']) ** 2 + (data_agg['sum_diff_y_micron']) ** 2)
# distance from first to last coordinate
data_agg['f_persistence'] = data_agg['f_total_displacement'] / data_agg['f_sum_diff_xy_micron']
# shows how directional the movement is

data_agg['file_mean_diff_xy_micron'] = data_agg.groupby('file')['f_mean_diff_xy_micron'].transform(np.max)
data_agg['f_fastest_mask'] = np.where((data_agg['f_mean_diff_xy_micron'] == data_agg['file_mean_diff_xy_micron']), 1, 0)
# DO NOT USE FOR guide AS TARGET (telo!)
# the fastest (or the only available) dot in the nucleus is 1, the rest is 0

data_agg['f_min_dist_range'] = data_agg['max_min_dist_micron']-data_agg['min_min_dist_micron']
# min_dist change within timelapse (max-min) for each dot
data_agg['f_total_min_dist'] = data_agg['end_min_dist_micron']-data_agg['beg_min_dist_micron']
# how distance changed within timelapse (frame29-frame0)

data_agg['file_max_min_dist_micron'] = data_agg.groupby('file')['f_min_dist_micron'].transform(np.max)
data_agg['f_most_central_mask'] = np.where((data_agg['f_min_dist_micron'] == data_agg['file_max_min_dist_micron']), 1, 0)
# DO NOT USE FOR guide AS TARGET (telo!)
# the most central (or the only available) dot in the nucleus is 1, the rest is 0

''' stratified cross-val K fold '''
