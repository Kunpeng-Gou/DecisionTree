import pandas as pd
from matplotlib import pyplot as plt


def data_read():
    return pd.read_csv('diabetes.csv')


if __name__ == '__main__':
    print('finished')
