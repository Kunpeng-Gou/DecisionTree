import data_processing as dp
import numpy as np
import pandas as pd
from numpy import log2


class Node(object):
    def __init__(self, id_list, feature=None, label=None, feature_val=None):
        self.id_list = id_list    # 储存这个结点中数据的id
        self.feature = feature    # 这个结点用来判别的特征
        self.label = label    # 如果是叶结点，则表示分类结果
        self.feature_val = feature_val    # 特征的分割值
        self.child = {}    # 子结点

    # 判断结点是否为叶结点
    def is_leaf(self):
        return self.child == {}

    # 根据返回值判断到达哪一个子结点
    def feature_ret(self, test):
        return test[self.feature] < self.feature_val

    def predict(self, test):
        if self.is_leaf():
            return self.label
        else:
            return self.child[self.feature_ret(test)].predict(test)


class DecisionTree(object):
    def __init__(self, train_data, train_id_list, algor=None, thresh=(0.01, 10), thresh_val=0.5):
        self.algor = algor    # 算法名称
        self.thresh = thresh    # 前剪枝的一些阈值
        self.thresh_val = thresh_val    # 决定分类的阈值
        self.train_data = train_data    # 训练集
        self.feature_list = train_data.columns    # 特征名称
        self.root = Node(train_id_list)    # 根结点

    def choose_best(self, node):
        ent = self.entropy(node.id_list)
        #print(ent)
        best_feature = None
        global_best_slip = 0
        global_best_ret = np.inf
        for feature in self.feature_list:
            if feature in ['Outcome', 'SkinThickness', 'Insulin']:
                break

            values = list({self.train_data[feature][i] for i in node.id_list} - {-1})
            values = sorted(values)
            slips = [(values[i] + values[i+1]) / 2 for i in range(len(values)-1)]

            local_best_slip = 0
            local_best_ret = np.inf
            for slip in slips:
                divided_list = [[], []]
                for i in node.id_list:
                    if self.train_data[feature][i] < slip and self.train_data[feature][i] != -1:
                        divided_list[0].append(i)
                    elif self.train_data[feature][i] > slip:
                        divided_list[1].append(i)
                # print(divided_list)
                num = np.array([len(divided_list[i]) for i in range(2)])
                # print(feature, num[0] + num[1])
                num = np.array([len(divided_list[i]) for i in range(2)]) / len(node.id_list)
                con_ent = np.array([self.entropy(divided_list[i]) for i in range(2)])
                # print(num, con_ent)
                ret = np.dot(num, con_ent.T)
                # print(ret)
                if ret < local_best_ret:
                    local_best_slip = slip
                    local_best_ret = ret
                    # print(ret, slip)

            if local_best_ret < global_best_ret:
                global_best_ret = local_best_ret
                global_best_slip = local_best_slip
                best_feature = feature
                # print(best_feature, global_best_slip)

        #print(best_feature, global_best_slip)
        return best_feature, global_best_slip, ent - global_best_ret

    def build_tree(self, node=None):
        if not node:
            node = self.root

        # 如果所有数据的标签相同，则将结点的标签设为该标签，并返回
        outcome = {self.train_data['Outcome'][i] for i in node.id_list}
        if len(outcome) == 1:
            node.label = outcome.pop()
            # print(node.label)
            return node

        # 选择最佳分类特征和分类特征的值
        best = self.choose_best(node)
        gain = best[2]
        # print(gain)
        # input()
        if gain < self.thresh_val:
            outcomes = [self.train_data['Outcome'][i] for i in node.id_list]
            cnt = pd.Series([0 for i in range(len(outcome))], index=list(outcome))
            for label in outcomes:
                cnt[label] += 1
            # print(cnt, cnt.idxmax())
            node.label = cnt.idxmax()
            # print(node.label)
            return node

        node.feature = feature = best[0]
        node.feature_val = slip = best[1]
        divided_list = [[], []]
        for i in node.id_list:
            if self.train_data[feature][i] < slip and self.train_data[feature][i] != -1:
                divided_list[0].append(i)
            elif self.train_data[feature][i] > slip:
                divided_list[1].append(i)
        num = np.array([len(divided_list[i]) for i in range(2)])
        # print(feature, num[0], num[1])
        # input()
        node.child[True] = Node(divided_list[0])
        self.build_tree(node.child[True])
        node.child[False] = Node(divided_list[1])
        self.build_tree(node.child[False])

        # print('finish tree building')
        # print(self.root.child, node.child)
        return node

    def entropy(self, id_list, feature='Outcome', feature_val=0.5):
        num = len(id_list)
        count = np.zeros(2)
        for i in id_list:
            if self.train_data[feature][i] < feature_val:
                count[0] += 1
            elif self.train_data[feature][i] > feature_val:
                count[1] += 1
        if count[0] * count[1] == 0:
            return 0
        count /= num
        # print(count)
        return -(count[0]*log2(count[0]) + count[1]*log2(count[1]))

    def predict(self, test):
        # print("DT predict")
        return self.root.predict(test)


class ID3(DecisionTree):
    def __init__(self, train_data, train_id_list, algor='ID3', thresh=(0.0001, 8), thresh_val=0.05,):
        super(ID3, self).__init__(train_data, train_id_list)
        self.algor = algor    # 算法名称
        self.thresh = thresh  # 前剪枝的一些阈值
        self.thresh_val = thresh_val  # 决定分类的阈值


if __name__ == '__main__':
    data = dp.data_read()
    dp.data_process(data)
    length = len(data.values)
    tree = ID3(data, range(length), thresh_val=0.025)
    # tree.predict()
    # print(tree.algor)
    # print(tree.train_data)
    tree.build_tree()
    # print(tree.root.child)
    # input()

    count = 0
    for i in range(length):
        print(tree.predict(data.loc[i]))
        if tree.predict(data.loc[i]) == data.loc[i]['Outcome']:
            count += 1
    print(count / length)

    print('finished')
