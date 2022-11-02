from to_become_public.feature_engineering import get_data  # todo: correct before publishing
import pandas as pd

path = 'to_become_public/tracking_output/data_47091baa.csv'  # todo: correct before publishing
data_from_csv = pd.read_csv(path)

frame_counts = data_from_csv.set_index(['file', 'particle']).index.value_counts()
less_30frames = frame_counts[frame_counts < 30]
data = data_from_csv.set_index(['file', 'particle']).drop(less_30frames.index)

X, y, indexed = get_data(data.reset_index())
