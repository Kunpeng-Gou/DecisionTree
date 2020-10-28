import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def data_read():
    return pd.read_csv('diabetes.csv')


def data_process(data):
    # 根据分析，对以下三个特征的空缺值用平均值进行填充
    fill_list = ['Glucose', 'BloodPressure', 'BMI']

    for feature in fill_list:
        zeros = dict(data[feature].value_counts())
        num = len(data.values) - zeros[0]
        avg = data[feature].sum() / num

        if feature == 'BMI':
            avg = round(avg, 1)
        else:
            avg = int(avg)

        data.loc[data[feature] == 0, feature] = avg

    # 以下特征的空缺值由0替换为-1（或者说替换为一个不会出现在正常情况下且不与其他特征的值冲突的值）
    replace_list = ['SkinThickness', 'Insulin']
    for feature in replace_list:
        data.loc[data[feature] == 0, feature] = -1


def zero_count(data):
    features = data.columns
    print(features)

    for feature in features:
        print(feature)
        zeros = dict(data[feature].value_counts())
        if 0 in zeros.keys():
            print(zeros[0])
        else:
            print(0)


# 将数据集分为k个不相交的子集，每个子集中的两种标签的数据数量相等
def cross_validation(data, k):
    count = data['Outcome'].value_counts()
    for i in range(len(count)):
        count[i] = int(count[i] / k)

    # random_list里储存结果为0，1的id列表，再分别打乱这些id
    random_list = [data[data.Outcome == 0].index.tolist(), data[data.Outcome == 1].index.tolist()]
    for id_list in random_list:
        random.shuffle(id_list)

    # 前k-1份的数量是相同的
    ret = []
    for i in range(k - 1):
        temp = set(random_list[0][i*count[0]:(i+1)*count[0]]) | set(random_list[1][i*count[1]:(i+1)*count[1]])
        ret.append(list(temp))

    # 最后一份处理k不整除count的情况
    temp = set(random_list[0][(k-1)*count[0]:]) | set(random_list[1][(k-1)*count[1]:])
    ret.append(temp)
    return ret


if __name__ == '__main__':
    da = data_read()
    print(da.columns)

    # zero_count(da)
    data_process(da)
    # zero_count(da)

    cross_validation(da, 10)
    print('finished')
