import pandas as pd
from matplotlib import pyplot as plt


def data_read():
    return pd.read_csv('diabetes.csv')


def data_process(data):
    #根据分析，对一下三个特征的空缺值用平均值进行填充
    fill_list = ['Glucose', 'BloodPressure', 'BMI']

    for feature in fill_list:
        #print(feature)
        zeros = dict(data[feature].value_counts())
        num = len(data.values) - zeros[0]
        avg = data[feature].sum() / num

        if feature == 'BMI':
            avg = round(avg, 1)
        else:
            avg = int(avg)
        print(avg)

        data.loc[data[feature] == 0, feature] = avg
        #print(dict(data[feature].value_counts()))


def zero_count(data):
    features = data.columns
    print(features)

    for feature in features:
        print(feature)
        zeros = dict(data[feature].value_counts())
        #print(zeros)
        if 0 in zeros.keys():
            print(zeros[0])
        else:
            print(0)


if __name__ == '__main__':
    da = data_read()
    #print(da.info())

    #zero_count(da)
    data_process(da)
    zero_count(da)
    print('finished')
