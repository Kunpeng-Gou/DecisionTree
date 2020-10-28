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
        self.child = {}    # 子结点（储存为字典，将featest_ret的值映射到子结点）

    # 判断结点是否为叶结点
    def is_leaf(self):
        return self.child == {}

    # 根据返回值判断到达哪一个子结点
    def feature_ret(self, test):
        return test[self.feature] < self.feature_val

    # 进行预测
    def predict(self, test):
        if self.is_leaf():
            return self.label
        else:
            return self.child[self.feature_ret(test)].predict(test)


class DecisionTree(object):
    def __init__(self, data, train_id_list, algor=None, thresh=(0.01, 10), thresh_val=0.5):
        self.algor = algor    # 算法名称
        self.thresh = thresh    # 前剪枝的一些阈值
        self.thresh_val = thresh_val    # 决定分类的阈值
        self.data = data    # 训练集
        self.feature_list = data.columns    # 特征名称
        self.root = Node(train_id_list)    # 根结点
        self.id_weight = None    # 暂时还未用到

    # 返回最佳分类特征，最佳分割点，对应的信息增益或信息增益比
    def choose_best(self, node):
        best_feature = None
        global_best_slip = 0
        global_best_ret = np.inf
        return best_feature, global_best_slip, global_best_ret

    # 建树
    def build_tree(self, node=None):
        if not node:
            node = self.root

        # 如果所有数据的标签相同，则将结点的标签设为该标签，并返回
        outcome = {self.data['Outcome'][i] for i in node.id_list}
        if len(outcome) == 1:
            node.label = outcome.pop()
            return node

        # 如果结点里储存的数据数量小于某个值，就直接选取标签数量最多的标签作为分类标签
        if len(node.id_list) < 5:
            outcomes = [self.data['Outcome'][i] for i in node.id_list]
            cnt = pd.Series([0 for i in range(len(outcome))], index=list(outcome))
            for label in outcomes:
                cnt[label] += 1
            node.label = cnt.idxmax()
            return node

        # 选择最佳分类特征和分类特征的值
        best = self.choose_best(node)
        ret = best[2]
        # 判断结果是否小于阈值
        if ret < self.thresh_val and self.algor != 'CART':
            outcomes = [self.data['Outcome'][i] for i in node.id_list]
            cnt = pd.Series([0 for i in range(len(outcome))], index=list(outcome))
            for label in outcomes:
                cnt[label] += 1
            node.label = cnt.idxmax()
            return node

        node.feature = feature = best[0]
        node.feature_val = slip = best[1]
        divided_list = [[], []]
        for j in node.id_list:
            if self.data[feature][j] < slip:
                if self.data[feature][j] != -1:
                    divided_list[0].append(j)
            elif self.data[feature][j] > slip:
                divided_list[1].append(j)
        num = np.array([len(divided_list[i]) for i in range(2)])
        node.child[True] = Node(divided_list[0])
        self.build_tree(node.child[True])
        node.child[False] = Node(divided_list[1])
        self.build_tree(node.child[False])

        return node

    # 计算信息熵
    def entropy(self, id_list, feature='Outcome', feature_val=0.5):
        num = len(id_list)
        count = np.zeros(2)
        for i in id_list:
            # 或者说'Outcome'=0
            if self.data[feature][i] < feature_val:
                count[0] += 1
            # 或者说'Outcome'=1
            elif self.data[feature][i] > feature_val:
                count[1] += 1
        if count[0] * count[1] == 0:
            return 0
        count /= num
        return -(count[0]*log2(count[0]) + count[1]*log2(count[1]))

    # 计算基尼指数
    def gini(self, id_list, feature='Outcome', feature_val=0.5):
        num = len(id_list)
        count = np.zeros(2)
        for i in id_list:
            # 或者说'Outcome'=0
            if self.data[feature][i] < feature_val:
                count[0] += 1
            # 或者说'Outcome'=1
            elif self.data[feature][i] > feature_val:
                count[1] += 1
        if count[0] * count[1] == 0:
            return 0
        count /= num

        return -(count[0] * log2(count[0]) + count[1] * log2(count[1]))

    # 进行预测
    def predict(self, test):
        return self.root.predict(test)


class ID3(DecisionTree):
    def __init__(self, data, train_id_list, algor='ID3', thresh=(0.0001, 8), thresh_val=0.5, ):
        super(ID3, self).__init__(data, train_id_list)
        self.algor = algor    # 算法名称
        self.thresh = thresh  # 前剪枝的一些阈值
        self.thresh_val = thresh_val  # 决定分类的阈值

    def choose_best(self, node):
        ent = self.entropy(node.id_list)

        best_feature = None  # 全局最佳特征
        global_best_slip = 0  # 全局最佳分割点
        global_best_ret = np.inf  # 全局最佳信息增益

        for feature in self.feature_list:
            # ID3无法处理有缺失值的数据，顺便忽视Outcome
            if feature in ['Outcome', 'SkinThickness', 'Insulin']:
                break

            # 按照二分法对连续值进行处理
            values = list({self.data[feature][i] for i in node.id_list} - {-1})
            values = sorted(values)
            slips = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]

            local_best_slip = 0  # 局部最佳分割点
            local_best_ret = np.inf  # 局部最佳信息增益

            # 计算每个分割点的信息增益
            for slip in slips:
                divided_list = [[], []]
                for i in node.id_list:
                    if self.data[feature][i] < slip and self.data[feature][i] != -1:
                        divided_list[0].append(i)
                    elif self.data[feature][i] > slip:
                        divided_list[1].append(i)
                num = np.array([len(divided_list[i]) for i in range(2)]) / len(node.id_list)
                con_ent = np.array([self.entropy(divided_list[i]) for i in range(2)])
                ret = np.dot(num, con_ent.T)

                # 更新局部最佳
                if ret < local_best_ret:
                    local_best_slip = slip
                    local_best_ret = ret

            # 更新全局最佳
            if local_best_ret < global_best_ret:
                global_best_ret = local_best_ret
                global_best_slip = local_best_slip
                best_feature = feature

        return best_feature, global_best_slip, ent - global_best_ret


class C45(DecisionTree):
    def __init__(self, data, train_id_list, algor='C4_5', thresh=(0.0001, 8), thresh_val=0.5, ):
        super(C45, self).__init__(data, train_id_list)
        self.algor = algor  # 算法名称
        self.thresh = thresh  # 前剪枝的一些阈值
        self.thresh_val = thresh_val  # 决定分类的阈值

        # 返回最佳特征，对应分割点，对应信息增益比

    def choose_best(self, node):
        print('C45')
        ent = self.entropy(node.id_list)
        best_feature = None  # 全局最佳特征
        global_best_slip = 0  # 全局最佳分割点
        global_best_ret = 0  # 全局最佳信息增益比
        for feature in self.feature_list:
            # 暂时先把这些缺失值较多的数据舍弃，顺便忽视Outcome
            if feature in ['Outcome', 'SkinThickness', 'Insulin']:
                break

            # 按照二分法对连续值进行处理
            values = list({self.data[feature][i] for i in node.id_list} - {-1})
            values = sorted(values)
            slips = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]

            local_best_slip = 0  # 局部最佳分割点
            local_best_ret = 0  # 局部最佳信息增益比

            # 计算每个分割点的信息增益
            for slip in slips:
                divided_list = [[], []]
                for i in node.id_list:
                    if self.data[feature][i] < slip and self.data[feature][i] != -1:
                        divided_list[0].append(i)
                    elif self.data[feature][i] > slip:
                        divided_list[1].append(i)
                # num = np.array([len(divided_list[i]) for i in range(2)])
                # print(num)
                num = np.array([len(divided_list[i]) for i in range(2)]) / len(node.id_list)
                con_ent = np.array([self.entropy(divided_list[i]) for i in range(2)])
                ret = np.dot(num, con_ent.T)

                fea_ent = -np.dot(num, log2(num).T)
                ret /= fea_ent
                # 更新局部最佳
                if ret > local_best_ret:
                    local_best_slip = slip
                    local_best_ret = ret

            # 更新全局最佳
            if local_best_ret > global_best_ret:
                global_best_ret = local_best_ret
                global_best_slip = local_best_slip
                best_feature = feature

        return best_feature, global_best_slip, global_best_ret


class CART(DecisionTree):
    def __init__(self, data, train_id_list, algor='CART', thresh=(0.0001, 8), thresh_val=0.5, ):
        super(CART, self).__init__(data, train_id_list)
        self.algor = algor  # 算法名称
        self.thresh = thresh  # 前剪枝的一些阈值
        self.thresh_val = thresh_val  # 决定分类的阈值

    def choose_best(self, node):
        best_feature = None  # 全局最佳特征
        global_best_slip = 0  # 全局最佳分割点
        global_best_ret = np.inf  # 全局最佳基尼指数
        for feature in self.feature_list:
            # 暂时先把这些缺失值较多的数据舍弃，顺便忽视Outcome
            if feature in ['Outcome', 'SkinThickness', 'Insulin']:
                break

            # 按照二分法对连续值进行处理
            values = list({self.data[feature][i] for i in node.id_list} - {-1})
            values = sorted(values)
            slips = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]

            local_best_slip = 0  # 局部最佳分割点
            local_best_ret = np.inf  # 局部最佳信息增益

            # 计算每个分割点的基尼指数
            for slip in slips:
                divided_list = [[], []]
                for i in node.id_list:
                    if self.data[feature][i] < slip and self.data[feature][i] != -1:
                        divided_list[0].append(i)
                    elif self.data[feature][i] > slip:
                        divided_list[1].append(i)
                num = np.array([len(divided_list[i]) for i in range(2)]) / len(node.id_list)
                gini_index = np.array([self.gini(divided_list[i]) for i in range(2)])
                ret = np.dot(num, gini_index.T)

                # 更新局部最佳
                if ret < local_best_ret:
                    local_best_slip = slip
                    local_best_ret = ret

            # 更新全局最佳
            if local_best_ret < global_best_ret:
                global_best_ret = local_best_ret
                global_best_slip = local_best_slip
                best_feature = feature

        return best_feature, global_best_slip, global_best_ret


if __name__ == '__main__':
    da = dp.data_read()
    dp.data_process(da)
    k = 20
    k_da = dp.cross_validation(da, k)

    const = set(list(range(len(da.values))))
    for j in range(k):
        train_id = list(const - set(k_da[j]))
        tree = CART(da, train_id, thresh_val=0.05)
        tree.build_tree()
        count = 0
        length = len(k_da[j])
        for i in k_da[j]:
            # print(tree.predict(da.loc[i]))
            if tree.predict(da.loc[i]) == da.loc[i]['Outcome']:
                count += 1
        print(count / length)

    print('finished')
