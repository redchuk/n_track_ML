# this is auxiliary script to check which directionality is right for raw data (inceptiontime)
import pandas as pd
import numpy as np
import math

path = 'to_become_public/tracking_output/data_47091baa.csv'
data = pd.read_csv(path)

data['dX'] = data['x_micron'].diff()
data['dY'] = data['y_micron'].diff()
data['t'] = data['dY'] / data['dX']

'''
data['dt'] = data['t'].diff()
data['dt_abs'] = data['dt'].abs()

print(data['dt_abs'].max())
print(data['dt_abs'].mean())
# tan(theta) = dy/dx seems a bad choice for directionality estimation, +/- infinity, if dx is close to 0

data['theta'] = np.arctan(data['t'])
# arctan() doesn't work either, since it ignores actually direction 
# (dy=1, dx=1) and (dy=-1, dx=-1) have the same arctan()
'''

test_data = pd.DataFrame({'theta_f0': [45, 45, 45, 45, -135, -135, -135, -135],
                          'theta_f1': [0, 90, -45, 135, 0, 90, 150, -60]})
test_data_radian = test_data * math.pi / 180  # to radians as in np trigonometry
test_data_radian['d_theta']=test_data_radian['theta_f1'] - test_data_radian['theta_f0']

