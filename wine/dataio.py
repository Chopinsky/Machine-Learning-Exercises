import pandas as pd
import os


def get_data_sets():
    curr_path = os.path.dirname(os.path.abspath(__file__))

    remote_white = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality  \
                    /winequality-white.csv'
    remote_red = 'http://archive.ics.uci.edu/ml/machine-learning-databases \
                    /wine-quality/winequality-red.csv'

    local_white = curr_path + '\\data\\white.csv'
    local_red = curr_path + '\\data\\red.csv'

    return remote_white, remote_red, local_white, local_red


def load_wine_data(local, link):
    if local == '' or link == '':
        return None

    with open(local, 'r') as inFile:
        data = pd.read_csv(inFile, sep=';')

    if len(data) == 0:
        print('loading data from link')
        data = pd.read_csv(link, sep=';')
    else:
        print('loading data from file')

    return data
