import pandas as pd


# reading the data
data = pd.read_csv('scripts/data_chromatin_live.csv')

# add features
data_agg = data.groupby(['file', 'particle']).agg(guide=('guide', 'first'),
                                                  time=('time', 'first'),
                                                  serum_conc_percent=('serum_conc_percent', 'first'),
                                                  mean_diff_xy_micron=('diff_xy_micron', 'mean'),
                                                  max_diff_xy_micron=('diff_xy_micron', 'max'),
                                                  sum_diff_xy_micron=('diff_xy_micron', 'sum'),
                                                  )



# stratified cross-val K fold

