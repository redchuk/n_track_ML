import pandas as pd
import dabest

data = pd.read_csv('scripts/data_sterile_449e453.csv')

def gen_set(data, parameter, chr_name, guides):
    '''

    :param data: df
    :param parameter: feature to plot, str
    :param chr_name: chr name, for labeling
    :param guides: guides to aggregate
    :return: df for Dabest plotting
    '''

    data = data[data["t_guide"].str.contains('1398') | data["t_guide"].str.contains('1514')]
