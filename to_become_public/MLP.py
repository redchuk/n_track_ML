from to_become_public.feature_engineering import get_data  # todo: correct before publishing
import pandas as pd

path = 'to_become_public/tracking_output/data_47091baa.csv'
indexed = get_data(path)[2]

frame_counts = pd.read_csv(path).set_index(['file', 'particle']).index.value_counts()
less_30frames = frame_counts[frame_counts < 30]
indexed = indexed.set_index(['file', 'particle']).drop(less_30frames.index)
#todo: drop <30frames for GBC?



