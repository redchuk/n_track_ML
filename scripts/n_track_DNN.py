import pandas as pd
import numpy as np

''' read the data '''
data = pd.read_csv('scripts/data_chromatin_live.csv')

''' add features '''
data_agg = data.groupby(['file', 'particle']).agg(guide=('guide', 'first'),
                                                  time=('time', 'first'),
                                                  serum_conc_percent=('serum_conc_percent', 'first'),

                                                  mean_diff_xy_micron=('diff_xy_micron', 'mean'),
                                                  # average displacement
                                                  max_diff_xy_micron=('diff_xy_micron', 'max'),  # maximal displacement
                                                  sum_diff_xy_micron=('diff_xy_micron', 'sum'),
                                                  # total trajectory length

                                                  sum_diff_x_micron=('diff_x_micron', 'sum'),
                                                  sum_diff_y_micron=('diff_y_micron', 'sum'),

                                                  )

data_agg['total_displacement'] = np.sqrt((data_agg['sum_diff_x_micron'])**2 + (data_agg['sum_diff_y_micron'])**2)
# distance from first to last coordinate
data_agg['persistence'] = data_agg['total_displacement']/data_agg['sum_diff_xy_micron']
# shows how directional the movement is


'''
example
df1 = df.groupby("b").mean().cumsum()
print (df1)
   a
b   
1  2
2  5

df['a'] = df['b'].map(df1['a'])
print (df)
   a  b
0  2  1
1  2  1
2  5  2
3  5  2

more example  df['sum_values_A'] = df.groupby('A')['values'].transform(np.sum)
'''


data_agg['f_mean_diff_xy_micron'] = data_agg.groupby('file')['mean_diff_xy_micron'].transform(np.max)
# this is for masking then (fastest dot)




''' stratified cross-val K fold '''
