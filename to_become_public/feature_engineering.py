import pandas as pd


data = pd.read_csv('to_become_public/tracking_output/data_47091baa.csv')

data['dX'] = data['x_micron'].diff()
data['dY'] = data['y_micron'].diff()

data_agg = data.groupby(['file', 'particle']).agg(serum=('serum', 'first'),

                                                  MD=('D', 'mean'),
                                                  # Mean locus displacement
                                                  MaxD=('D', 'max'),
                                                  # Maximal locus displacement
                                                  temp_sum_D=('D', 'sum'),
                                                  VarD=('D', 'var'),
                                                  # Variance of locus displacement

                                                  temp_sum_dX=('dX', 'sum'),
                                                  temp_sum_dY=('dY', 'sum'),

                                                  MA=('A', 'mean'),
                                                  # Mean nuclear area
                                                  MP=('P', 'mean'),
                                                  # Mean nuclear perimeter

                                                  MDist=('Dist', 'mean'),
                                                  # Mean locus distance to nuclear periphery
                                                  temp_min_Dist=('Dist', 'min'),
                                                  temp_max_Dist=('Dist', 'max'),
                                                  temp_beg_Dist=('Dist', 'first'),
                                                  temp_end_Dist=('Dist', 'last'),
                                                  VarDist=('Dist', 'var'),
                                                  # Variance of locus distance to nuclear periphery
                                                  )

data_agg['f_Rvar_diff_xy_micron'] = data_agg['VarD'] / data_agg['f_mean_diff_xy_micron']
data_agg['f_Rvar_dist_micron'] = data_agg['VarDist'] / data_agg['MDist']
# Relative variance

data_agg['f_total_displacement'] = np.sqrt((data_agg['temp_sum_dX']) ** 2 + (data_agg['temp_sum_dY']) ** 2)
# distance from first to last coordinate
data_agg['f_persistence'] = data_agg['f_total_displacement'] / data_agg['temp_sum_D']
# shows how directional the movement is

data_agg['file_mean_diff_xy_micron'] = data_agg.groupby('file')['f_mean_diff_xy_micron'].transform(np.max)
data_agg['f_fastestemp_mask'] = np.where((data_agg['f_mean_diff_xy_micron'] == data_agg['file_mean_diff_xy_micron']), 1, 0)
# DO NOT USE FOR guide AS TARGET (telo!)
# the fastest (or the only available) dot in the nucleus is 1, the rest is 0

data_agg['f_min_dist_range'] = data_agg['temp_max_Dist'] - data_agg['temp_min_Dist']
# min_dist change within timelapse (max-min) for each dot
data_agg['f_total_min_dist'] = data_agg['temp_end_Dist'] - data_agg['temp_beg_Dist']
# how distance changed within timelapse (frame29-frame0)

data_agg['file_max_min_dist_micron'] = data_agg.groupby('file')['MDist'].transform(np.max)
data_agg['f_most_central_mask'] = np.where((data_agg['MDist'] == data_agg['file_max_min_dist_micron']), 1,
                                           0)
# DO NOT USE FOR guide AS TARGET (telo!)
# the most central (or the only available) dot in the nucleus is 1, the rest is 0

data_slope = data.groupby(['file', 'particle']).apply(lambda x: linregress(x['frame'], x['Dist'])[0])
data_agg['f_slope_min_dist_micron'] = data_slope
# slope for minimal distance to edge; how distance to edge changes within the timelapse?


data_slope_area = data.groupby(['file', 'particle']).apply(lambda x: linregress(x['frame'], x['area_micron'])[0])
data_agg['f_slope_area_micron'] = data_slope_area
# slope for nucleus area; how area changes within the timelapse?

data_slope_perimeter = data.groupby(['file', 'particle']).apply(lambda x: linregress(x['frame'],
                                                                                     x['perimeter_au_norm'])[0])
data_agg['f_slope_perimeter_au_norm'] = data_slope_perimeter
# slope for nucleus perimeter

data_SD_diff_xy_micron = data.groupby(['file', 'particle']).agg(SD_diff=('D', 'std'))
data_i = data.set_index(['file', 'particle'])
data_i['SD_diff_xy_micron'] = data_SD_diff_xy_micron
data_i['f_mean_diff_xy_micron'] = data_agg['f_mean_diff_xy_micron']
data_i['outliers2SD_diff_xy'] = np.where((data_i['D'] >
                                          (data_i['f_mean_diff_xy_micron'] + 2 * data_i['SD_diff_xy_micron'])), 1, 0)
data_i['outliers3SD_diff_xy'] = np.where((data_i['D'] >
                                          (data_i['f_mean_diff_xy_micron'] + 3 * data_i['SD_diff_xy_micron'])), 1, 0)
data_agg['f_outliers2SD_diff_xy'] = data_i.groupby(['file', 'particle']) \
    .agg(f_outliers2SD_diff_xy=('outliers2SD_diff_xy', 'sum'))
data_agg['f_outliers3SD_diff_xy'] = data_i.groupby(['file', 'particle']) \
    .agg(f_outliers3SD_diff_xy=('outliers3SD_diff_xy', 'sum'))
# is there a displacement larger than mean plus 2SD or 3SD (SD calculated for each dot, 29xy pairs) respectively

data_sterile = data_agg.drop(['temp_sum_dX',
                              'temp_sum_dY',
                              'temp_min_Dist',
                              'temp_max_Dist',
                              'temp_beg_Dist',
                              'temp_end_Dist',
                              'file_mean_diff_xy_micron',
                              'file_max_min_dist_micron',
                              'temp_sum_D',
                              ], axis=1)
data_sterile.reset_index(inplace=True)